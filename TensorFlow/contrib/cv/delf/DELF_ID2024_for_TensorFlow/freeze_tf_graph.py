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
"""Freeze graph script for DELF/G model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os
import time
import itertools
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
tf.disable_eager_execution()
import numpy as np
from tqdm import tqdm

from datasets.google_landmarks_dataset import googlelandmarks as gld
from datasets.revisited_op import dataset
from model import delf_model


FLAGS = flags.FLAGS
flags.DEFINE_string('output_path', '/tmp/delf', 'WithTensorBoard output_path.')
flags.DEFINE_string('data_path', '/tmp/data', 'data path.')
flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
flags.DEFINE_boolean('block3_strides', True, 'Whether to use block3_strides.')
flags.DEFINE_boolean('use_augmentation', True,
                     'Whether to use ImageNet style augmentation.')
flags.DEFINE_integer('image_size', 321, 'Size of each image side to use.')
flags.DEFINE_integer('topk', 20, 'topk result.')
flags.DEFINE_boolean('use_autoencoder', True,
                     'Whether to train an autoencoder.')
flags.DEFINE_integer(
    'autoencoder_dimensions', 128,
    'Number of dimensions of the autoencoder. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_integer(
    'local_feature_map_channels', 1024,
    'Number of channels at backbone layer used for local feature extraction. '
    'Default value 1024 is the number of channels of block3. Used only if'
    'use_autoencoder=True.')


def eval_step(model, image):
  images = tf.expand_dims(image, axis=0)

  # Make a forward pass to calculate prelogits.
  (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
    dim_expanded_features, dim_reduced_features) = model.global_and_local_forward_pass(images)

  return attn_scores, dim_reduced_features, desc_prelogits


def read_and_decode(example, image_size=321):
    feature_description = {
        'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = gld.NormalizeImages(image, pixel_value_scale=128.0, pixel_value_offset=128.0)
    image = tf.image.resize(image, [image_size, image_size])
    image.set_shape([image_size, image_size, 3])
    return image


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('data_path= %s', FLAGS.data_path)  # data_path/best_ckpts: checkpoint directory 
  logging.info('output_path= %s', FLAGS.output_path)
  pb_model_dir = os.path.join(FLAGS.output_path, 'pb_model')  # pb output directory

  image_size = FLAGS.image_size
  tf.reset_default_graph()
  # Setup session
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config_proto.allow_soft_placement = True
  tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))

  model = delf_model.Delf(
      block3_strides=FLAGS.block3_strides,
      name='DELF',
      use_dim_reduction=FLAGS.use_autoencoder,
      reduced_dimension=FLAGS.autoencoder_dimensions,
      dim_expand_channels=FLAGS.local_feature_map_channels)
  # model.init_classifiers(model.num_classes)
  image_data = tf.placeholder(tf.float32, shape=(321, 321, 3))
  attn_scores, val_features, val_global_features = eval_step(model, image_data)

  # restore from best ckpt 
  saver = tf.train.Saver(max_to_keep=3)
  model_dir = os.path.join(FLAGS.data_path, "best_ckpts")

  global_step_value = 0
  global_step = tf.train.get_or_create_global_step()

  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  saver.restore(tfs, latest_ckpt)
  logging.info("Restore from ckpt:{}".format(latest_ckpt))

  # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
  # for tensor_name in tensor_name_list:
  #   if "autoenc_conv1" in tensor_name:
  #     print(tensor_name)
  frozen_gd = tf.graph_util.convert_variables_to_constants(
      tfs, tf.get_default_graph().as_graph_def(), ['autoencoder/autoenc_conv1/BiasAdd'])
  tf.io.write_graph(frozen_gd, pb_model_dir, "delf_model.pb", as_text=False)
  logging.info("Export to delf_model.pb in {}.".format(pb_model_dir))


if __name__ == '__main__':
  app.run(main)
