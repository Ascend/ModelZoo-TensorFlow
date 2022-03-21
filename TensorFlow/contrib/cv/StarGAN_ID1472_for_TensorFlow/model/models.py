# -*- coding:utf-8 -*-
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
Generator and Discriminator network.
"""
import tensorflow as tf
import numpy as np
from model.ops import conv, deconv, flatten, resblock, lrelu, relu, tanh, instance_norm

def network_Gen(name, in_data, c, num_filters, g_n_blocks=6, reuse=False):
    """
    Generator network.
    :param name:
    :param in_data:
    :param c:
    :param num_filters:
    :param g_n_blocks:
    :param reuse:
    :return:
    """
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        c = tf.tile(c, [1, in_data.shape[1], in_data.shape[2], 1])
        x = tf.concat([in_data, c], axis=-1)

        channel = num_filters

        x = conv(x, channel, kernel=7, stride=1, pad=3, use_bias=False, scope='conv')
        x = instance_norm(x, scope='ins_norm')
        x = relu(x)

        # Down-Sampling
        for i in range(2):
            x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_' + str(i))
            x = instance_norm(x, scope='down_ins_norm_' + str(i))
            x = relu(x)

            channel = channel * 2

        # Bottleneck
        for i in range(g_n_blocks):
            x = resblock(x, channel, use_bias=False, scope='resblock_' + str(i))

        # Up-Sampling
        for i in range(2):
            x = deconv(x, channel // 2, kernel=4, stride=2, use_bias=False, scope='deconv_' + str(i))
            x = instance_norm(x, scope='up_ins_norm' + str(i))
            x = relu(x)

            channel = channel // 2

        x = conv(x, channels=3, kernel=7, stride=1, pad=3, use_bias=False, scope='G_logit')
        x = tanh(x)

        return x


def network_Dis(name, in_data, image_size, num_filters, c_dim, d_n_blocks=6, reuse=False):
    """
    Discriminator network.
    :param name:
    :param in_data:
    :param image_size:
    :param num_filters:
    :param c_dim:
    :param d_n_blocks:
    :param reuse:
    :return:
    """
    assert in_data is not None
    with tf.variable_scope(name, reuse=reuse):
        channel = num_filters
        x = conv(in_data, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
        x = lrelu(x, 0.01)

        for i in range(1, d_n_blocks):
            x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
            x = lrelu(x, 0.01)
            channel = channel * 2

        c_kernel = int(image_size / np.power(2, d_n_blocks))

        logit = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')

        c = conv(x, channels=c_dim, kernel=c_kernel, stride=1, use_bias=False, scope='D_label')
        c = tf.reshape(c, shape=[-1, c_dim])

        return logit, c
