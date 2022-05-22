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

from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np


def conv_layer(bottom, kernel_size, in_channel, out_channel, stride, name, padding="VALID"):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape=[kernel_size, kernel_size, in_channel,
                                        out_channel], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "bias", shape=[out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(bottom, w, strides=[
            1, stride, stride, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)
        return conv


def fc_layer(bottom, out_dims, name):
    bottom = tf.reshape(bottom, shape=[-1, bottom.get_shape().as_list()[-1]])
    in_dims = bottom.shape[-1].value
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape=[
            in_dims, out_dims], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "bias", shape=[out_dims], initializer=tf.constant_initializer(0.0))
        print()
        fc = tf.nn.bias_add(tf.matmul(bottom, w), b)
        return fc


def fc_layer_v1(bottom, in_dims, out_dims, name):
    bottom = tf.reshape(bottom, shape=[-1, bottom.get_shape().as_list()[-1]])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("weights", shape=[
            in_dims, out_dims], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(
            "bias", shape=[out_dims], initializer=tf.constant_initializer(0.0))
        print()
        fc = tf.nn.bias_add(tf.matmul(bottom, w), b)
        return fc


def bn(inputTensor, is_training, name):
    # _BATCH_NORM_DECAY = 0.99
    # _BATCH_NORM_EPSILON = 1E-12
    return tf.layers.batch_normalization(inputTensor, training=is_training, name=name)


def avgpool(bottom, kernel_size=2, stride=2, name="avg"):
    return tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                          padding="VALID", name=name)


def maxpool(bottom, kernel_size=2, stride=2, name="max"):
    return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],
                          padding="VALID", name=name)


def res_block_3_layer(bottom, channel_list, name, change_dimension=False, block_stride=1, is_training=True):
    with tf.variable_scope(name) as scope:
        if change_dimension:
            short_cut_conv = conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2], block_stride,
                                        "shortcut")
            block_conv_input = bn(short_cut_conv, is_training, name="shortcut")
        else:
            block_conv_input = bottom
        block_conv1 = conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride,
                                 "a")
        block_conv1 = bn(block_conv1, is_training, name="a")
        block_conv1 = tf.nn.relu(block_conv1)
        block_conv2 = conv_layer(block_conv1, 3, channel_list[0], channel_list[1], 1, "b", padding="SAME")
        block_conv2 = bn(block_conv2, is_training, name="b")
        block_conv2 = tf.nn.relu(block_conv2)
        block_conv3 = conv_layer(block_conv2, 1, channel_list[1], channel_list[2], 1, "c")
        block_conv3 = bn(block_conv3, is_training, name="c")

        block_res = tf.add(block_conv_input, block_conv3)
        relu = tf.nn.relu(block_res)
        return relu

