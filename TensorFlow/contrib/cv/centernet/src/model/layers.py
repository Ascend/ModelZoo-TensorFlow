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

import tensorflow as tf


def conv2d_bn_relu(inputs,
                   filters,
                   kernel_size,
                   strides=1,
                   padding='valid',
                   conv=tf.layers.conv2d,
                   use_bn=True,
                   is_training=False,
                   activation=tf.nn.relu):
    outputs = conv(inputs,
                   filters,
                   kernel_size,
                   strides,
                   padding,
                   use_bias=not use_bn)
    if use_bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


def basic_block(inputs,
                filters,
                strides,
                conv=tf.layers.conv2d,
                use_bn=True,
                is_training=False):
    conv1 = conv2d_bn_relu(inputs,
                           filters,
                           kernel_size=(3, 3),
                           strides=strides,
                           padding='same',
                           conv=conv,
                           use_bn=use_bn,
                           is_training=is_training)
    conv2 = conv2d_bn_relu(conv1,
                           filters,
                           kernel_size=(3, 3),
                           strides=1,
                           padding='same',
                           conv=conv,
                           use_bn=use_bn,
                           is_training=is_training,
                           activation=None)
    if strides != 1 or tf.shape(inputs)[-1] != filters:
        inputs = conv2d_bn_relu(inputs,
                                filters,
                                kernel_size=(1, 1),
                                strides=strides,
                                padding='same',
                                conv=conv,
                                use_bn=use_bn,
                                is_training=is_training,
                                activation=None)
    outputs = tf.nn.relu(conv2 + inputs)
    return outputs
