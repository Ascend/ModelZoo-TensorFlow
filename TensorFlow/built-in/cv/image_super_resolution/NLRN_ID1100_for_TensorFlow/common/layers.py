#
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
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf


class Conv2DWeightNorm(tf.layers.Conv2D):

  def build(self, input_shape):
    self.wn_g = self.add_weight(
        name='wn_g',
        shape=(self.filters,),
        dtype=self.dtype,
        initializer=tf.initializers.ones,
        trainable=True,
    )
    super(Conv2DWeightNorm, self).build(input_shape)
    square_sum = tf.reduce_sum(
        tf.square(self.kernel), [0, 1, 2], keepdims=False)
    inv_norm = tf.rsqrt(square_sum)
    self.kernel = self.kernel * (inv_norm * self.wn_g)


def conv2d_weight_norm(inputs,
                       filters,
                       kernel_size,
                       strides=(1, 1),
                       padding='valid',
                       data_format='channels_last',
                       dilation_rate=(1, 1),
                       activation=None,
                       use_bias=True,
                       kernel_initializer=None,
                       bias_initializer=tf.zeros_initializer(),
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       kernel_constraint=None,
                       bias_constraint=None,
                       trainable=True,
                       name=None,
                       reuse=None):
  layer = Conv2DWeightNorm(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)
