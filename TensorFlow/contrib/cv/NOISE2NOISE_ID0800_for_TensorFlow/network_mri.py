# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# ============================================================================
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
# ============================================================================
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
import numpy as np


# The network.

def autoencoder(input):
    def conv(n, name, n_out, size=3, gain=np.sqrt(2)):
        with tf.variable_scope(name):
            wshape = [size, size, int(n.get_shape()[-1]), n_out]
            wstd = gain / np.sqrt(np.prod(wshape[:-1]))  # He init
            W = tf.get_variable('W', shape=wshape, initializer=tf.initializers.random_normal(0., wstd))
            b = tf.get_variable('b', shape=[n_out], initializer=tf.initializers.zeros())
            n = tf.nn.conv2d(n, W, strides=[1] * 4, padding='SAME')
            n = tf.nn.bias_add(n, b)
        return n

    def tile(x, multiples):
        x_list = [x]
        for i in range(len(multiples)):
            x = multiples[i] * x_list
            x_list = [tf.concat(x, axis=i)]
        return x_list[0]

    def up(n, name, f=2):
        with tf.name_scope(name):
            s = [-1 if i.value is None else i.value for i in n.shape]
            n = tf.reshape(n, [s[0], s[1], 1, s[2], 1, s[3]])
            # n = tf.tile(n, [1, 1, f, 1, f, 1])
            n = tile(n, [1, 1, f, 1, f, 1])
            n = tf.reshape(n, [s[0], s[1] * f, s[2] * f, s[3]])
        return n

    def down(n, name, f=2):     return tf.nn.max_pool(n, ksize=[1, f, f, 1], strides=[1, f, f, 1], padding='SAME',
                                                      name=name)

    def concat(name, layers):   return tf.concat(layers, axis=-1, name=name)

    def LR(n):                  return tf.nn.leaky_relu(n, alpha=0.1, name='lrelu')

    # Make even size and add the channel dimension.

    input = tf.pad(input, ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=-.5)
    input = tf.expand_dims(input, axis=-1)

    # Encoder part.

    x = input
    x = LR(conv(x, 'enc_conv0', 48))
    x = LR(conv(x, 'enc_conv1', 48))
    x = down(x, 'pool1')
    pool1 = x

    x = LR(conv(x, 'enc_conv2', 48))
    x = down(x, 'pool2')
    pool2 = x

    x = LR(conv(x, 'enc_conv3', 48))
    x = down(x, 'pool3')
    pool3 = x

    x = LR(conv(x, 'enc_conv4', 48))
    x = down(x, 'pool4')
    pool4 = x

    x = LR(conv(x, 'enc_conv5', 48))
    x = down(x, 'pool5')

    x = LR(conv(x, 'enc_conv6', 48))

    # Decoder part.

    x = up(x, 'upsample5')
    x = concat('concat5', [x, pool4])
    x = LR(conv(x, 'dec_conv5', 96))
    x = LR(conv(x, 'dec_conv5b', 96))

    x = up(x, 'upsample4')
    x = concat('concat4', [x, pool3])
    x = LR(conv(x, 'dec_conv4', 96))
    x = LR(conv(x, 'dec_conv4b', 96))

    x = up(x, 'upsample3')
    x = concat('concat3', [x, pool2])
    x = LR(conv(x, 'dec_conv3', 96))
    x = LR(conv(x, 'dec_conv3b', 96))

    x = up(x, 'upsample2')
    x = concat('concat2', [x, pool1])
    x = LR(conv(x, 'dec_conv2', 96))
    x = LR(conv(x, 'dec_conv2b', 96))

    x = up(x, 'upsample1')
    x = concat('concat1', [x, input])
    x = LR(conv(x, 'dec_conv1a', 64))
    x = LR(conv(x, 'dec_conv1b', 32))

    x = conv(x, 'dec_conv1', 1, gain=1.0)

    # Remove the channel dimension and crop to odd size.

    return tf.squeeze(x, axis=-1)[:, :-1, :-1]
