"""
tools
"""
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf
import gman_flags as df


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    # with tf.device('/cpu:0'):
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv(layer_name, x, in_channels, out_channels, kernel_size=None, stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        kernel_size:
        stride:

    Returns:

    """
    # dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = _variable_with_weight_decay(name='weights',
                                        shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                        stddev=5e-2, wd=0.0)
        b = _variable_on_cpu(name='biases', shape=[out_channels],
                             initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def conv_eval(layer_name, x, in_channels, out_channels, kernel_size=None, stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        kernel_size:
        stride:

    Returns:

    """
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = _variable_with_weight_decay(name='weights',
                                        shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                        stddev=5e-2, wd=0.0)
        b = _variable_on_cpu(name='biases', shape=[out_channels],
                             initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def conv_nonacti(layer_name, x, in_channels, out_channels, kernel_size=None, stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        kernel_size:
        stride:

    Returns:

    """
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights_nonacti', dtype=dtype,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases_nonacti', dtype=dtype,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv_nonacti')
        x = tf.nn.bias_add(x, b, name='bias_add_nonacti')
        return x


def conv_nonacti_eval(layer_name, x, in_channels, out_channels, kernel_size=None, stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        kernel_size:
        stride:

    Returns:

    """
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights_nonacti', dtype=dtype,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='biases_nonacti', dtype=dtype,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv_nonacti')
        x = tf.nn.bias_add(x, b, name='bias_add_nonacti')
        return x


def acti_layer(x):
    """

    Args:
        x:

    Returns:

    """
    x = tf.nn.relu(x, name='only_relu')
    return x


def deconv(layer_name, x, in_channels, out_channels, output_shape=None, kernel_size=None,
           stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        output_shape:
        kernel_size:
        stride:

    Returns:

    """
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    if output_shape is None:
        output_shape = [32, 224, 224, 64]
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights', dtype=dtype,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        # b = tf.get_variable(name='biases', dtype=dtype,
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='SAME',
                                   name='deconv')
        # x = tf.nn.bias_add(x, b, name='bias_add')
        # x = tf.nn.relu(x, name='relu')
        return x


def deconv_eval(layer_name, x, in_channels, out_channels, output_shape=None, kernel_size=None,
                stride=None):
    """

    Args:
        layer_name:
        x:
        in_channels:
        out_channels:
        output_shape:
        kernel_size:
        stride:

    Returns:

    """
    if stride is None:
        stride = [1, 1, 1, 1]
    if kernel_size is None:
        kernel_size = [3, 3]
    if output_shape is None:
        output_shape = [32, 224, 224, 64]
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name='weights', dtype=dtype,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        # b = tf.get_variable(name='biases',
        #                     shape=[out_channels],
        #                     initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='SAME',
                                   name='deconv')
        # x = tf.nn.bias_add(x, b, name='bias_add')
        # x = tf.nn.relu(x, name='relu')
        return x


def pool(layer_name, x, kernel=None, stride=None, is_max_pool=True):
    """

    Args:
        layer_name:
        x:
        kernel:
        stride:
        is_max_pool:

    Returns:

    """
    if stride is None:
        stride = [1, 2, 2, 1]
    if kernel is None:
        kernel = [1, 2, 2, 1]
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def batch_norm(x):
    """

    Args:
        x:

    Returns:

    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def norm(name, x, lsize=4):
    """

    Args:
        name:
        x:
        lsize:

    Returns:

    """
    tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    return x


def FC_layer(layer_name, x, out_nodes):
    """

    Args:
        layer_name:
        x:
        out_nodes:

    Returns:

    """
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights', dtype=dtype,
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases', dtype=dtype,
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


def weight(kernel_shape):
    """

    Args:
        kernel_shape:
        is_uniform:

    Returns:

    """
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    w = tf.get_variable(name='weights', dtype=dtype,
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


def bias(bias_shape):
    """

    Args:
        bias_shape:

    Returns:

    """
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    b = tf.get_variable(name='biases', dtype=dtype,
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b


if __name__ == '__main__':
    pass
