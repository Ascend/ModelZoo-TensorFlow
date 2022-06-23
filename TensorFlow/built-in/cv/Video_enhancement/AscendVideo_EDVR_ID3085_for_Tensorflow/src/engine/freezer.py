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

import os

import numpy as np
import tensorflow as tf
from src.runner.saver import loose_loading
from src.runner.sess_config import get_sess_config
from src.utils.adapter import NetworkIOAdapter
from src.utils.logger import logger
from tensorflow.python.framework import graph_util


class _Freezer:
    """Freezer engine to freeze ckpt to pb file.

    Args:
        dataloader: None, will never be used. The input size will be determined
            by configure.
        network: network instance, will not be used in this mode.
        cfg: yacs node, global configuration.
    """
    def __init__(self, dataloader, network, cfg, **kwargs):
        self.cfg = cfg
        self.network = network
        # self.adapter = NetworkIOAdapter(cfg)   # do not register_raw_size. Use the setting value

        # Different from the inference, we'll fix the input size.
        # The fixed input size is given by:
        #   cfg.data.inference.best_patch_size[0] + pads_h + cfg.data.inference.patch_pad_size[0]
        #   cfg.data.inference.best_patch_size[1] + pads_w + cfg.data.inference.patch_pad_size[1]
        # pads_h, pads_w = self.adapter.cal_adapted_size(self.adapter.best_in_size)
        # self.adapter.limited_in_size = [self.adapter.best_in_size[0] + pads_h + self.adapter.eval_pad_size*2,
        #                                 self.adapter.best_in_size[1] + pads_w + self.adapter.eval_pad_size*2]
        # self.adapter.register_raw_size(self.adapter.limited_in_size)

        self.network.build_graph(input_size=(cfg.data.inference.batch_size, (None, None)))

    def restore(self):
        """
        Restore the graph from ckpt.
        """
        raise NotImplementedError

    def run(self):
        """
        Execute function to freeze the graph to pb.
        """
        raise NotImplementedError


class SessionFreezer(_Freezer):
    """
    A tf.Session based freezer engine.
    """
    def __init__(self, dataloader, network, cfg):
        super().__init__(dataloader, network, cfg)
        sess_cfg = get_sess_config(cfg.env.device,
                                   cfg.session.xla,
                                   cfg.session.mix_precision,
                                   False)
        self.session = tf.Session(config=sess_cfg)

    def restore(self):
        """
        Restore the requireed part of the graph given the ckpt.
        """
        loose_loading(self.session, self.cfg.model.scope, '', self.cfg.checkpoint)

    def run(self):
        """
        Execute function to freeze the graph to pb.
        """
        with self.session as sess:
            tf.io.write_graph(sess.graph_def, self.cfg.checkpoint.rsplit('/', 1)[0], 'freeze_graph.pbtxt')

            logger.info('Loading trained model ...')
            self.restore()
            logger.info('Model loaded success.')
            logger.info('Freeze model to pb files')

            pb_path = os.path.join(self.cfg.checkpoint + '.pb')
            try:
                if hasattr(self.network, 'inference_func'):
                    constant_graph = graph_util.convert_variables_to_constants(
                        sess, sess.graph_def,
                        self.network.output_node_name
                    )
                else:
                    constant_graph = graph_util.convert_variables_to_constants(
                        sess, sess.graph_def,
                        [self.network.output_node.name.split(':')[0]]
                    )
                with tf.gfile.FastGFile(pb_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                logger.info('Model frozen success.')
            except Exception as e:
                logger.error('Failed to freeze model.')
                logger.info(e)
