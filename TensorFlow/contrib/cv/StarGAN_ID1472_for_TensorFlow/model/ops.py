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

import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    '''

    :param x:
    :param channels:
    :param kernel:
    :param stride:
    :param pad:
    :param pad_type:
    :param use_bias:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv_0'):
    '''

    :param x:
    :param channels:
    :param kernel:
    :param stride:
    :param use_bias:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x,
                                       filters=channels,
                                       kernel_size=kernel,
                                       kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride,
                                       padding='SAME',
                                       use_bias=use_bias)

        return x


def flatten(x):
    '''

    :param x:
    :return:
    '''
    return tf.layers.flatten(x)


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    '''

    :param x_init:
    :param channels:
    :param use_bias:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    '''

    :param x:
    :param alpha:
    :return:
    '''
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    '''

    :param x:
    :return:
    '''
    return tf.nn.relu(x)


def tanh(x):
    '''

    :param x:
    :return:
    '''
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    '''

    :param x:
    :param scope:
    :return:
    '''
    x = tf_contrib.layers.instance_norm(x,
                                        epsilon=1e-05,
                                        center=True, 
                                        scale=True,
                                        scope=scope)
    return x

