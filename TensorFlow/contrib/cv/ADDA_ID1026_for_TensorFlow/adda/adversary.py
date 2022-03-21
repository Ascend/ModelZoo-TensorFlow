"""
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
"""

from contextlib import ExitStack

import tensorflow as tf
import tflearn
from tensorflow.contrib import slim


def adversarial_discriminator(net, layers, scope='adversary', leaky=False):
    if leaky:
        activation_fn = tflearn.activations.leaky_relu
    else:
        activation_fn = tf.nn.relu
    with ExitStack() as stack:
        stack.enter_context(tf.variable_scope(scope))
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected],
                activation_fn=activation_fn,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))
        for dim in layers:
            net = slim.fully_connected(net, dim)
        net = slim.fully_connected(net, 2, activation_fn=None)
    return net
