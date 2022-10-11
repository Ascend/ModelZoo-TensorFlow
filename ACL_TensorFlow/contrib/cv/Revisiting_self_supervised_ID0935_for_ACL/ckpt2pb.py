#!/usr/bin/python
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
"""The main script for starting training and evaluation.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
import numpy as np
from npu_bridge.npu_init import *
import functools
import os
import absl.app as app
import absl.flags as flags
import absl.logging as logging
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
from self_supervision.self_supervision_lib import get_self_supervision_model

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_file', "log/ckpt_file", 'the file stored ckpt file')
flags.DEFINE_string('pb_file', "log/pb_file", 'the file to store pb model.')
flags.DEFINE_string('frozen_pb_file', "log/frozen_pb_file", 'the file to store frozen pb model.')
flags.DEFINE_string('task', "supervised", 'Which pretext-task to learn from. Can be '
                                          'one of `rotation`, `exemplar`, `jigsaw`, '
                                          '`relative_patch_location`, `linear_eval`, `supervised`.')
flags.mark_flag_as_required('task')

# Flags about the model.
flags.DEFINE_string('architecture', "resnet50", help='Which basic network architecture to use. '
                                                     'One of vgg19, resnet50, revnet50.')
flags.DEFINE_integer('filters_factor', 4, 'Widening factor for network '
                                          'filters. For ResNet, default = 4 = vanilla ResNet.')
flags.DEFINE_bool('last_relu', None, 'Whether to include (default) the final '
                                     'ReLU layer in ResNet/RevNet models or not.')
flags.DEFINE_string('mode', None, 'Which ResNet to use, `v1` or `v2`.')

flags.DEFINE_float('weight_decay', 1e-4, 'Strength of weight-decay. '
                                         'Defaults to 1e-4, and may be set to 0.')


def get_model():
    # export the pb module from ckpt module
    estimator = tf.estimator.Estimator(model_fn=get_self_supervision_model(FLAGS.task), )
    checkpoint = tf.train.get_checkpoint_state(FLAGS.ckpt_file)
    checkpoint_path = checkpoint.model_checkpoint_path
    input_ids = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='input')
    data_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features={'image': input_ids, },
                                                                      default_batch_size=1)
    result_path = estimator.export_savedmodel(export_dir_base=FLAGS.pb_file,
                                              serving_input_receiver_fn=data_fn,
                                              checkpoint_path=checkpoint_path)
    # free the graph
    freeze_graph.freeze_graph(
        input_saved_model_dir=result_path,
        output_node_names='module/Squeeze_1',
        output_graph=os.path.join(FLAGS.frozen_pb_file, "frozen_model.pb"),
        initializer_nodes='',
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False)


def main(unused_argv):
    get_model()
    logging.info('ckpt to pb is success')


if __name__ == '__main__':
    app.run(main)
