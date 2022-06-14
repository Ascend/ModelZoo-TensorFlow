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

from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def circle_loss(y_true,
               y_pred,
               gamma = 256,
               margin = 0.25,
               batch_size=None):
  O_p = 1 + margin
  O_n = -margin
  Delta_p = 1 - margin
  Delta_n = margin
  if batch_size:
    batch_size = batch_size
    batch_idxs = tf.expand_dims(
  tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]
  alpha_p = tf.nn.relu(O_p - tf.stop_gradient(y_pred))
  alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - O_n)
  # yapf: disable
  y_true = tf.cast(y_true, tf.float32)
  y_pred = (y_true * (alpha_p * (y_pred - Delta_p)) +
            (1 - y_true) * (alpha_n * (y_pred - Delta_n))) * gamma
  # yapf: enable
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

