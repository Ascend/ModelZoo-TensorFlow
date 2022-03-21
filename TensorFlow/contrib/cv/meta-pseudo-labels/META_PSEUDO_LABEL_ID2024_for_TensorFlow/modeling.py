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

import sys

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
import modeling_utils as ops


class Wrn28k(object):
  """WideResNet."""

  def __init__(self, params, k=2):
    self.params = params
    self.name = f'wrn-28-{k}'
    self.k = k
    logging.info(f'Build `wrn-28-{k}` under scope `{self.name}`')

  def __call__(self, x, training, start_core_index=0, final_core_index=1):
    if training:
      logging.info(f'Call {self.name} for `training`')
    else:
      logging.info(f'Call {self.name} for `eval`')

    params = self.params
    k = self.k
    
    ### TODO: significant bug
    s = [16, 135, 135*2, 135*4] if k == 135 else [16*k, 16*k, 32*k, 64*k]
    # s = [16, 135, 135*2, 135*4] if k == 135 else [16, 16*k, 32*k, 64*k]

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      with tf.variable_scope('stem'):
        x = ops.conv2d(x, 3, s[0], 1)
        ops.log_tensor(x, True)

      x = ops.wrn_block(x, params, s[1], 1, training, 'block_1')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_2')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_3')
      x = ops.wrn_block(x, params, s[1], 1, training, 'block_4')

      x = ops.wrn_block(x, params, s[2], 2, training, 'block_5')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_6')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_7')
      x = ops.wrn_block(x, params, s[2], 1, training, 'block_8')

      x = ops.wrn_block(x, params, s[3], 2, training, 'block_9')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_10')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_11')
      x = ops.wrn_block(x, params, s[3], 1, training, 'block_12')

      with tf.variable_scope('head'):
        x = ops.batch_norm(x, params, training)
        ### TODO: x = ops.gpu_batch_norm(x, params, training)
        x = ops.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
        ops.log_tensor(x, True)
        
        x = ops.dropout(x, params.dense_dropout_rate, training)
        x = ops.dense(x, params.num_classes)
        x = tf.cast(x, dtype=tf.float32, name='logits')
        ops.log_tensor(x, True)

    return x

