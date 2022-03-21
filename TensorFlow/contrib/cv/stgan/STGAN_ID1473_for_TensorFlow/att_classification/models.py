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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tflib as tl
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_regularizer=slim.l2_regularizer(1e-4))
fc = partial(tl.flatten_fully_connected, activation_fn=None, weights_regularizer=slim.l2_regularizer(1e-4))
relu = tf.nn.relu
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
pool = partial(slim.max_pool2d, kernel_size=2, stride=2)


def classifier(x, att_dim=40, dim=32, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('classifier', reuse=reuse):
#    with tf.name_scope('classifier'):
        y = conv_bn_relu(x, dim / 2, 3, 1)
        y = conv_bn_relu(y, dim / 2, 3, 1)
        y = pool(y)
        y = conv_bn_relu(y, dim * 1, 3, 1)
        y = conv_bn_relu(y, dim * 1, 3, 1)
        y = pool(y)
        y = conv_bn_relu(y, dim * 2, 3, 1)
        y = conv_bn_relu(y, dim * 2, 3, 1)
        y = pool(y)
        y = conv_bn_relu(y, dim * 4, 3, 1)
        y = conv_bn_relu(y, dim * 4, 3, 1)
        y = pool(y)
        y = conv_bn_relu(y, dim * 8, 3, 1)
        y = conv_bn_relu(y, dim * 8, 3, 1)
        y = pool(y)
        y = relu(fc(y, dim * 16))
        logits = fc(y, att_dim)
        return logits
