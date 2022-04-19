# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import shutil
import time
from functools import partial
from multiprocessing import Manager

import numpy as np
import tensorflow as tf
from src.runner.saver import loose_loading
from src.runner.sess_config import get_sess_config
from src.utils.adapter import NetworkIOAdapter
from src.utils.exceptions import *
from src.utils.file_io import ImageWriter, image_deprocess
from src.utils.logger import logger
from src.utils.moving_avg import MovingAvg
from src.utils.constant import FILE_EXT_TO_PIX_FMT
from src.utils.world import world
from tqdm import trange


class _Inferencer:
    """Base inference engine.

    Args:
        dataloader: dict, inference data dict produced by test dataset instance.
        network: network instance, whose class should derive from 
            src.networks.base_model.Base .
        cfg: yacs node, global configuration.
        _world: world instance, could be given from the caller function,
            or by default, the global world instance (see src.utils.world).
    """
    def __init__(self, dataloader, network, cfg, _world=None):
        self.device = cfg.env.device
        self.is_distributed = cfg.env.rank_size > 1
        self.cfg = cfg
        self.dataloader = dataloader
        self.network = network
        self.step_time = MovingAvg(smooth=0.9)
        self.adapter = NetworkIOAdapter(cfg)
        # _world should be initialized
        self.world = _world or world
        if not self.world.is_initialized:
            raise WorldUninitializedError('World not initialized.')

        self._total = 0

    def restore(self, *args, **kwargs):
        """ 
        Restore parameters from ckpt.
        """
        raise NotImplementedError

    def run(self):
        """
        Execute inference steps.
        """
        raise NotImplementedError


# Note: we use numpy dataloader instead of tf dataloader in inference
class SessionInferencer(_Inferencer):
    """Session based inference engine.

    Args:
        dataloader: dict, inference data dict produced by test dataset instance.
        network: network instance, whose class should derive from 
            src.networks.base_model.Base .
        cfg: yacs node, global configuration.
        _world: world instance, could be given from the caller function,
            or by default, the global world instance (see src.utils.world).
    """
    def __init__(self, dataloader, network, cfg, _world=None):
        super().__init__(dataloader, network, cfg, _world)
        self.scale = cfg.model.scale
        self.session = None
        self.graph = None

        # Get expected output data information. Both are used for ffmpeg io-backend.
        output_size = self.dataloader.expect_output_resolution
        output_ext = self.dataloader.expect_output_file_ext
        
        pix_fmt = FILE_EXT_TO_PIX_FMT[output_ext]

        # Prepare image writer if is set.
        if not self.cfg.inference.write_out:
            logger.warn(f'You have set "write_out" to False, '
                        f'hence there will be no outputs to {self.cfg.inference.io_backend}.')
        else:
            output_dir = self.cfg.inference.result_dir
            output_dir = os.path.realpath(output_dir)
            # By default, we write results to hard disk
            self.image_deprocess_fn = partial(
                image_deprocess,
                source_color_space=self.cfg.data.color_space,
                benormalized=self.cfg.data.normalized)
            self.result_writer = ImageWriter(
                output_dir, cfg,
                benormalized=self.cfg.data.normalized,
                source_color_space=self.cfg.data.color_space,
                output_resolution=output_size,
                pix_fmt=pix_fmt)

    def restore(self):
        """Restore parameters from ckpt.
        """
        if self.cfg.checkpoint == 'none':
            # Reserved for tasks that are not performed using networks.
            pass
        elif (self.cfg.checkpoint == '' 
              and len(self.cfg.train.pretrained_scope_list) > 0):
            # For models that consists of several sub-networks, e.g., vfi model
            # with pretrained optical flow network.
            assert len(self.cfg.train.pretrained_scope_list) == \
                   len(self.cfg.train.pretrained_scope_ckpt)
            for scope, ckpt in zip(self.cfg.train.pretrained_scope_list, 
                                   self.cfg.train.pretrained_scope_ckpt):
                loose_loading(self.session, scope, '', ckpt)
            return 0
        else:
            # Commonly used branch.
            return loose_loading(self.session, self.cfg.model.scope, 
                                 '', self.cfg.checkpoint)

    def network_preparation(self):
        """Build network forward graph, and restor from ckpt.
        """
        sess_cfg = get_sess_config(self.cfg.env.device,
                                   self.cfg.session.xla,
                                   self.cfg.session.mix_precision,
                                   False)
        self.session = tf.Session(config=sess_cfg)

        # Register the image raw size when inference to let the adapter decide
        # whether to inference using patchwise strategy or as a whole.
        self.adapter.register_raw_size(self.dataloader.raw_image_size)

        # Get the real adapted input size from adapter to build the graph.
        self.network.build_graph(input_size=(self.cfg.data.inference.batch_size, 
                                             self.adapter.input_size))
        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        self.session.run(init_op)
        if self.cfg.debug_mode != 'zeroin':
            self.restore()
    
    def run(self):
        """Execute inference steps.
        """
        self.network_preparation()

        # Dataset shard is done in building dataset, see dataloaders.__init__.py
        self._total = len(self.dataloader)

        range_fn = partial(trange, position=self.world.rank_id, desc=f'On DeviceID {self.world.device_id}')

        if self.session is None:
            raise SessionUndefinedError(f'{type(self).__name__}.session is not defined.')

        logger.info(f'Start inference.')

        if self.cfg.inference.write_out:
            self.result_writer.initialize()

        for i in range_fn(self._total):
            data_dict = self.dataloader[i]
            st_time = time.time()
            hq = self._inf_engine(data_dict)
            once_time = time.time() - st_time
            # Skip the first step since the elapse time is abnormal due to compilation.
            if i > 0:
                self.step_time.update(once_time)

            if self.cfg.inference.write_out:
                self.write_out(data_dict['output_file'], hq, data_dict.get('input_copies', None))

        if self.cfg.inference.write_out:
            self.result_writer.finalize()
        logger.info(f'\tInference time: {self.step_time.avg * 1000:.2f} ms/image')

    def _inf_engine(self, data_dict):
        """Determine inference strategy.
        """
        # TODO: support multiple feed dict.
        lq = data_dict['lq']
        if hasattr(self.network, 'inference_func'):
            # Reserved API if the processing of the network is not end-to-end.
            # Pass through all the inputs, in case the model requires multiple-inputs.
            data_dict['lq'] = self.adapter.adapt_input(data_dict['lq'])
            hq = self.network.inference_func(self.session, data_dict, self.graph, self.adapter.mode)
            hq = self.adapter.reverse_adapt(hq.squeeze())
        elif self.adapter.patch_mode:
            patch_per_step = self.cfg.data.inference.batch_size
            img_patches = self.adapter.extract_image_patches(lq, patch_per_step)
            num_step = img_patches.shape[0] // patch_per_step
            patch_hq = []
            for i in range(num_step):
                batch_data = img_patches[i * patch_per_step:(i + 1) * patch_per_step]
                if patch_per_step == 1 and batch_data.shape[0] != 1 and self.cfg.model.input_format_dimension == 5:
                    batch_data = batch_data[None, ...]
                elif self.cfg.model.input_format_dimension == 4:
                    batch_data = np.reshape(batch_data, [-1, *batch_data.shape[2:]])
                _patch_hq = self._inf_func(batch_data)
                patch_hq.extend(_patch_hq)
            hq = self.adapter.stitching_patches_to_image(patch_hq)
        else:
            lq = self.adapter.adapt_input(lq)
            hq = self._inf_func(lq[None])
            hq = self.adapter.reverse_adapt(hq.squeeze())
        return hq.squeeze()

    def _inf_func(self, lq):
        """Real calling inference function.

        Args:
            lq: numpy array, input array.

        Returns:
            hq: numpy array, processd output array.
        """
        # TODO: support multiple feed dict.
        hq = self.session.run(self.network.output_node, feed_dict={self.network.input_node: lq})
        return hq

    def write_out(self, output_files, network_outputs, input_copies):
        """Write out function.
        """
        output_dict = dict()

        if isinstance(output_files, (list, tuple)):
            assert len(output_files) == len(network_outputs)
            network_outputs_ = [self.image_deprocess_fn(n, hdr=output_files[0].endswith('.exr'))
                                for n in network_outputs]
            output_dict.update(dict(zip(output_files, network_outputs_)))
        elif isinstance(output_files, str):
            network_outputs_ = self.image_deprocess_fn(network_outputs, hdr=output_files.endswith('.exr'))
            output_dict[output_files] = network_outputs_
        else:
            raise NotImplementedError

        if input_copies is not None:
            # deprocess the copied data
            input_copies_deprocess = dict()
            for k, v in input_copies.items():
                input_copies_deprocess[k] = [v[0], self.image_deprocess_fn(v[1], hdr=k.endswith('.exr'))]
            output_dict.update(input_copies_deprocess)

        self.result_writer.write_out(output_dict)


class ModelFreeInferencer(SessionInferencer):
    """
    Inferencer using pb file, without model python file.
    """
    def restore(self):
        """Restore from pb file.

        Returns:
            graph: tf.graph, the forward tensorflow graph.
        """
        with tf.gfile.GFile(self.cfg.checkpoint, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    def network_preparation(self):
        """Build network forward graph, and restor from pb. Prepare adapter.
        """
        sess_cfg = get_sess_config(self.cfg.env.device,
                                   self.cfg.solver.xla,
                                   self.cfg.solver.mix_precision,
                                   False)

        # Load from PB
        self.graph = self.restore()
        self.session = tf.Session(config=sess_cfg, graph=self.graph)

        # Fix the real eval in size before register image raw size.
        # This function will use the 
        #   model.best_in_size + data.eval_padsize * 2 
        # as the fixed eval in size
        self.adapter.fix_eval_in_size()
        self.adapter.register_raw_size(self.dataloader.raw_image_size)

    def _inf_func(self, lq):
        """Real calling inference function.

        Args:
            lq: numpy array, input array.

        Returns:
            hq: numpy array, processd output array.
        """
        hq = self.session.run(self.graph.get_tensor_by_name("SR_output:0"), 
                              feed_dict={self.graph.get_tensor_by_name("L_input:0"): lq})
        return hq
