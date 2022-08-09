# Lint as: python3
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
# ==============================================================================
"""Export DELF tensorflow inference model.

The exported model may use an image pyramid for multi-scale processing, with
local feature extraction including receptive field calculation and keypoint
selection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os

from absl import app
from absl import flags
import tensorflow as tf

from model import delf_model
from model import export_model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'ckpt_path', '/tmp/delf-logdir/delf-weights', 'Path to saved checkpoint.')
flags.DEFINE_string('export_path', None, 'Path where model will be exported.')
flags.DEFINE_boolean(
    'block3_strides', True, 'Whether to apply strides after block3.')
flags.DEFINE_float('iou', 1.0, 'IOU for non-max suppression.')
flags.DEFINE_boolean(
    'use_autoencoder', True,
    'Whether the exported model should use an autoencoder.')
flags.DEFINE_integer(
    'autoencoder_dimensions', 128,
    'Number of dimensions of the autoencoder. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_integer(
    'local_feature_map_channels', 1024,
    'Number of channels at backbone layer used for local feature extraction. '
    'Default value 1024 is the number of channels of block3. Used only if'
    'use_autoencoder=True.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  export_path = FLAGS.export_path
  if os.path.exists(export_path):
    raise ValueError(f'Export_path {export_path} already exists. Please '
                     'specify a different path or delete the existing one.')

  _checkpoint_path = FLAGS.ckpt_path
  _stride_factor = 2.0 if FLAGS.block3_strides else 1.0
  _iou = FLAGS.iou
  # Setup the DELF model for extraction.
  _model = delf_model.Delf(
      block3_strides=FLAGS.block3_strides,
      name='DELF',
      use_dim_reduction=FLAGS.use_autoencoder,
      reduced_dimension=FLAGS.autoencoder_dimensions,
      dim_expand_channels=FLAGS.local_feature_map_channels)

  input_image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
  input_scales = tf.placeholder(shape=[None], dtype=tf.float32, name='input_scales')
  input_max_feature_num = tf.placeholder(shape=(), dtype=tf.int32, name='input_max_feature_num')
  input_abs_thres = tf.placeholder(shape=(), dtype=tf.float32, name='input_abs_thres')
  input_tensors = {
          'input_image': input_image,
          'input_scales': input_scales,
          'input_max_feature_num': input_max_feature_num,
          'iput_abs_thres': input_abs_thres
          }

  # Setup session
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config_proto.allow_soft_placement = True
  # config_proto = tf.ConfigProto(device_count={'GPU':0})

  with tf.Session(config=npu_config_proto(config_proto=config_proto)) as sess:
    _model.load_weights(_checkpoint_path)
    print('Checkpoint loaded from ', _checkpoint_path)
    extracted_features = export_model_utils.ExtractLocalFeatures(
        input_image, input_scales, input_max_feature_num, input_abs_thres,
        _iou, lambda x: _model(x, training=False), _stride_factor)
    named_output_tensors = {}
    named_output_tensors['boxes'] = tf.identity(
        extracted_features[0], name='boxes')
    named_output_tensors['features'] = tf.identity(
        extracted_features[1], name='features')
    named_output_tensors['scales'] = tf.identity(
        extracted_features[2], name='scales')
    named_output_tensors['scores'] = tf.identity(
        extracted_features[3], name='scores')

    ### tf.saved_model.simple_save(
    ###     sess, export_path,
    ###     inputs=input_tensors,
    ###     outputs=named_output_tensors)


if __name__ == '__main__':
  app.run(main)

