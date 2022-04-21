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

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


# Concatenating 2 tensors
def concatenate(x, y):
    X_shape = x.get_shape()
    Y_shape = y.get_shape()
    # concatenating on feature map axis
    return tf.concat([x, y], axis=3)


# Define activation function for the network
def lrelu_layer(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# Function for fully connected layer
def linear_layer(x, output_size, scope=None, stddev=0.2, bias_start=0.0, with_w=False):
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:  # return values along with parameters of fc_layer
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


# Function for BatchNormalization layer
def bn_layer(x, is_training, scope):
    return layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training,
                             scope=scope)


# Function for 2D convolutional layer
def conv2d_layer(x, num_filters, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [filter_height, filter_width, x.get_shape()[-1], num_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))  # weights
        s = [1, stride_height, stride_width, 1]  # stride

        if name == 'res_convd1' or name == 'res_convd2':
            conv = tf.nn.conv2d(x, w, s, padding='SAME')
        else:
            conv = tf.nn.conv2d(x, w, s, padding='SAME')
            biases = tf.get_variable('bias', [num_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# Function for 2D Deconvolutional layer
def deconv2d_layer(x, out_channel, filter_height, filter_width, stride_height, stride_width, stddev=0.2,
                   name="deconv2d"):
    with tf.variable_scope(name):
        in_channel = x.get_shape()[-1]
        out_shape = [int(x.get_shape()[0]), int(x.get_shape()[1] * stride_height), int(x.get_shape()[2] * stride_width),
                     out_channel]
        # out_shape = tf.convert_to_tensor(out_shape)
        w = tf.get_variable("weight", [filter_height, filter_width, out_channel, x.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        s = [1, stride_height, stride_width, 1]
        deconv = tf.nn.conv2d_transpose(x, w, out_shape, s, padding='SAME')
        biases = tf.get_variable('bias', out_channel, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


# Function for Residual Blocks
def residual_block1(input, num_filters, filter_size, is_training, name="res_block"):
    with tf.variable_scope(name):
        x_shortcut = x
        x = lrelu_layer(bn_layer(conv2d_layer(x, num_filters, filter_size, filter_size, 2, 2, name='res_convd1'),
                                 is_training=is_training, scope='ebn_1'))
        x = bn_layer(conv2d_layer(x, num_filters, 1, 1, 1, 1, name='res_convd2'), is_training=is_training,
                     scope='ebn_2')
        x_shortcut = bn_layer(conv2d_layer(x_shortcut, num_filters, 1, 1, 1, 1, name='skip'), is_training=is_training,
                              scope='ebn_3')
        res = tf.nn.relu(x + x_shortcut)
        return res


# Function for Residual Blocks
def residual_block(input, num_filters, filter_size, is_training, name="res_block"):
    with tf.variable_scope(name):
        in_filter = input.get_shape()[-1]
        out_filter = num_filters
        x_shortcut = input
        x = lrelu_layer(bn_layer(conv2d_layer(input, out_filter, 1, 1, 2, 2, name='sub_res_1'), is_training=is_training,
                                 scope='bn_1'))  # 64 x 64 x 128
        x = lrelu_layer(bn_layer(conv2d_layer(x, out_filter, filter_size, filter_size, 1, 1, name='sub_res_2'),
                                 is_training=is_training, scope='bn_2'))  # 64 x 64 x 128
        x = bn_layer(conv2d_layer(x, out_filter, 1, 1, 1, 1, name='sub_res_3'), is_training=is_training,
                     scope='bn_3')  # 64 x 64 x 128
        x_shortcut = bn_layer(conv2d_layer(x_shortcut, out_filter, 1, 1, 2, 2, name='res_skip'),
                              is_training=is_training, scope='bn_skip')  # 64 x 64 x 128

        res = tf.nn.relu(x + x_shortcut)

        return res
