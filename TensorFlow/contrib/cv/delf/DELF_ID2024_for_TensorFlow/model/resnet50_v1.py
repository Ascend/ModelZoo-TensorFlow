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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=protected-access

r"""Models."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from npu_bridge.npu_init import *

import sys

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
import modeling_utils as ops


class ResNet50(object):
  """Bottleneck ResNet."""

  def __init__(self,
               data_format,
               name='',
               include_top=True,
               pooling=None,
               block3_strides=False,
               average_pooling=True,
               classes=1000,
               gem_power=3.0,
               embedding_layer=False,
               embedding_layer_dim=2048):
    self.name = name
    self.data_format = data_format
    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.include_top = include_top
    self.block3_strides = block3_strides
    self.average_pooling = average_pooling
    self.pooling = pooling
    self.embedding_layer_dim = embedding_layer_dim
    logging.info(f'Build `resnet-50` under scope `{self.name}`')

  def build__call(self, x, training=True, intermediates_dict=None):
    """Building the ResNet50 model.

    Args:
      x: Images to compute features for.
      training: Whether model is in training phase.
      intermediates_dict: `None` or dictionary. If not None, accumulate feature
        maps from intermediate blocks into the dictionary. ""

    Returns:
      Tensor with featuremap.
    """

    def _block_fn(inputs, num_out_filters, stride, name):
      return ops.resnet_block(inputs,
                              num_out_filters=num_out_filters,
                              stride=stride,
                              training=training,
                              name=name)

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      x = ops.conv2d(x, 7, 64, 2, data_format=self.data_format, padding='SAME', name='conv1')
      x = ops.batch_norm(x, training, name='bn_conv1')
      x = ops.relu(x, leaky=0.)
      if intermediates_dict is not None:
        intermediates_dict['block0'] = x

      x = ops.max_pool(x, 3, 2, data_format=self.data_format)
      if intermediates_dict is not None:
        intermediates_dict['block0mp'] = x

      # Block 1 (equivalent to "conv2" in Resnet paper).
      x = _block_fn(x, 256, 1, name='block1_a')
      x = _block_fn(x, 256, 1, name='block1_b')
      x = _block_fn(x, 256, 1, name='block1_c')
      if intermediates_dict is not None:
        intermediates_dict['block1'] = x

      # Block 2 (equivalent to "conv3" in Resnet paper).
      x = _block_fn(x, 512, 2, name='block2_a')
      x = _block_fn(x, 512, 1, name='block2_b')
      x = _block_fn(x, 512, 1, name='block2_c')
      x = _block_fn(x, 512, 1, name='block2_d')
      if intermediates_dict is not None:
        intermediates_dict['block2'] = x

      # Block 3 (equivalent to "conv4" in Resnet paper).
      x = _block_fn(x, 1024, 2, name='block3_a')
      x = _block_fn(x, 1024, 1, name='block3_b')
      x = _block_fn(x, 1024, 1, name='block3_c')
      x = _block_fn(x, 1024, 1, name='block3_d')
      x = _block_fn(x, 1024, 1, name='block3_e')
      x = _block_fn(x, 1024, 1, name='block3_f')

      if self.block3_strides:
        ### x = self.subsampling_layer(x)
        x = ops.max_pool(x, 1, 2, data_format=self.data_format)
        if intermediates_dict is not None:
          intermediates_dict['block3'] = x
        x = _block_fn(x, 2048, 1, name='block_a')
      else:
        if intermediates_dict is not None:
          intermediates_dict['block3'] = x
        x = _block_fn(x, 2048, 2, name='block_a')
      x = _block_fn(x, 2048, 1, name='block_b')
      x = _block_fn(x, 2048, 1, name='block_c')

      if self.average_pooling:
        x = ops.avg_pool(x, 7, 7, data_format=self.data_format)
        if intermediates_dict is not None:
          intermediates_dict['block4'] = x
      else:
        if intermediates_dict is not None:
          intermediates_dict['block4'] = x

      x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
      # x = ops.dropout(x, params.dense_dropout_rate, training)
      # x = ops.dense(x, self.embedding_layer_dim)  # embedding_layer is false
      # x = tf.cast(x, dtype=tf.float32, name='logits')

    return x

