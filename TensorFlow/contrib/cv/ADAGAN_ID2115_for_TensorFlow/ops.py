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
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Tensorflow ops used by GAN.

"""

import tensorflow as tf
import numpy as np
import logging

def lrelu(x, leak=0.3):
    return tf.maximum(x, leak * x)

def batch_norm(opts, _input, is_train, reuse, scope, scale=True):
    """Batch normalization based on tf.contrib.layers.

    """
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        epsilon=opts['batch_norm_eps'], decay=opts['batch_norm_decay'],
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def upsample_nn(input_, new_size, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "upsample_nn", reuse=reuse):
        result = tf.image.resize_nearest_neighbor(input_, new_size)

    return result

def downsample(input_, d_h=2, d_w=2, conv_filters_dim=None, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "downsample", reuse=reuse):
        result = tf.nn.max_pool(input_, ksize=[1, d_h, d_w, 1], strides=[1, d_h, d_w, 1], padding='SAME')

    return result

def linear(opts, input_, output_dim, scope=None, init='normal', reuse=None):
    """Fully connected linear layer.

    Args:
        input_: [num_points, ...] tensor, where every point can have an
            arbitrary shape. In case points are more than 1 dimensional,
            we will stretch them out in [numpoints, prod(dims)].
        output_dim: number of features for the output. I.e., the second
            dimensionality of the matrix W.
    """

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:
        # This means points contained in input_ have more than one
        # dimensions. In this case we first stretch them in one
        # dimensional vectors
        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal':
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        else:
            matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.constant_initializer(np.identity(in_shape)))
        bias = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias_start))


    return tf.matmul(input_, matrix) + bias


def conv2d(opts, input_, output_dim, d_h=2, d_w=2, scope=None,
           conv_filters_dim=None, padding='SAME', l2_norm=False):
    """Convolutional layer.

    Args:
        input_: should be a 4d tensor with [num_points, dim1, dim2, dim3].

    """

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(opts, input_, output_shape, d_h=2, d_w=2, scope=None, conv_filters_dim=None, padding='SAME'):
    """Transposed convolution (fractional stride convolution) layer.

    """

    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=[output_shape[0],int(output_shape[1]),int(output_shape[2]),int(output_shape[3])],
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)


    return deconv

def optimizer(opts, net=None, decay=1.):
    """Choose a suitable optimizer.

    """
    if net is not None:
        if net == 'g':
            learning_rate = opts['opt_g_learning_rate']
        elif net == 'd':
            learning_rate = opts['opt_d_learning_rate']
        else:
            assert False, 'Optimizer supports only d, g, or None modes.'
    else:
        learning_rate = opts['opt_learning_rate']
    learning_rate *= decay

    if opts["optimizer"] == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif opts["optimizer"] == "adam":
        return tf.train.AdamOptimizer(learning_rate, beta1=opts["opt_beta1"])
    else:
        assert False, 'Unknown optimizer.'

def log_sum_exp(logits):
    l_max = tf.reduce_max(logits, axis=1, keep_dims=True)
    return tf.add(l_max,
                  tf.reduce_sum(
                    tf.exp(tf.subtract(
                        logits,
                        tf.tile(l_max, tf.stack([1, logits.get_shape()[1]])))),
                    axis=1))
