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

"""Shake-shake and ShakeDrop utility functions."""
from npu_bridge.npu_init import *
import tensorflow as tf


def shake_shake_train(xa, xb, seed, name=None):
  """
  Shake-shake regularization: training.

  Shake-shake regularization interpolates between inputs A and B
  with *different* random uniform (per-sample) interpolation factors
  for the forward and backward/gradient passes

  Args:
    xa: input, branch A
    xb: input, branch B

  Returns:
    Mix of input branches
  """

  gate_shape = [xa.get_shape()[0], 1, 1, 1]
  gate_forward = tf.random.uniform(gate_shape, minval=0, maxval=1, seed=seed, name=''.join([name,'_gf']), dtype=tf.float32)
  gate_backward = tf.random.uniform(gate_shape, minval=0, maxval=1, seed=seed, name=''.join([name,'_gb']), dtype=tf.float32)
  x_forward = xa * gate_forward + xb * (1.0 - gate_forward)
  x_backward = xa * gate_backward + xb * (1.0 - gate_backward)
  x = x_backward + tf.stop_gradient(x_forward - x_backward)
  return x


def shake_shake_eval(xa, xb):
  """Shake-shake regularization: evaluation.

  Args:
    xa: input, branch A
    xb: input, branch B

  Returns:
    Mix of input branches
  """
  # Blend between inputs A and B 50%-50%.
  return (xa + xb) * 0.5


