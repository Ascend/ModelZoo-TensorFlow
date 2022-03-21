"""
ckpt_to_pb
"""
# coding=utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from npu_bridge.npu_init import RewriterConfig, npu_config_proto
import moxing as mx
import gman_constant as constant
import gman_flags as df
import gman_model as model
from tensorflow.python.framework import graph_util

if not os.path.exists(df.FLAGS.train_dir):
    os.makedirs(df.FLAGS.train_dir)
mx.file.copy_parallel(df.FLAGS.checkpoint_dir_obs, df.FLAGS.checkpoint_dir)


def ckpt2pb():
    """
    ckpt2pb
    Returns:

    """
    # A list used to save all psnr and ssim.
    psnr_list = []
    # Read all hazed images indexes and clear images from directory

    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # # set precision mode
        # custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
        #     'allow_mix_precision')
        # dump path
        # custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes('/cache/saveModels/')
        # # set dump debug
        # custom_op.parameter_map['enable_dump_debug'].b = True
        # custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        #     '{"output":"/cache/saveModels","task_trace":"on"}')
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # ########################################################################
        # ########################Load images from disk##############################
        # ########################################################################
        hazed_image_placeholder = tf.placeholder(tf.float32,
                                                 shape=[1, df.FLAGS.input_image_height, df.FLAGS.input_image_width,
                                                        constant.RGB_CHANNEL], name='feature')

        # ########################################################################
        # ###################Restore model and do evaluations#####################
        # ########################################################################
        gman = model.Gman()
        logist = gman.inference(hazed_image_placeholder, batch_size=1, h=df.FLAGS.input_image_height,
                                w=df.FLAGS.input_image_width)
        logist = tf.identity(logist, name='pred')
        variable_averages = tf.train.ExponentialMovingAverage(
            constant.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver, train_op, hazed_image, clear_image_arr, hazed_images_obj, placeholder, psnr_list, ssim_list, h, w
        with tf.Session(graph=graph, config=npu_config_proto(config_proto=sess_config)) as sess:
            ckpt = tf.train.get_checkpoint_state(df.FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
            graph = tf.get_default_graph()
            output_node_names = ['pred']
            input_graph_def = graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,  # = sess.graph_def,
                output_node_names=output_node_names)
            with tf.gfile.GFile('/cache/saveModels' + '/gman.pb', 'wb') as fgraph:
                fgraph.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    ckpt2pb()
    mx.file.copy_parallel('/cache/saveModels', df.FLAGS.train_url)
    print('copy saved results to obs: {}.'.format(df.FLAGS.train_url))
