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
Some codes from
https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def fc(inputs, num_out, name, activation_fn=None, reuse=None):
    shape = inputs.get_shape()
    if len(shape) == 4:
        inputs = tf.reshape(inputs, tf.stack([tf.shape(inputs)[0], np.prod(shape[1:])]))
        inputs.set_shape([None, np.prod(shape[1:])])

    w_init = tf.random_normal_initializer(stddev=0.02)

    return tf.contrib.layers.fully_connected(inputs, num_out, activation_fn=activation_fn, weights_initializer=w_init,
                                             reuse=reuse, scope=name)


def concat(inputs, axis):
    return tf.concat(values=inputs, axis=axis)


def conv_batch_normalization(inputs, name, epsilon=1e-5, is_training=True, activation_fn=None, reuse=None):
    return tf.contrib.layers.batch_norm(inputs, decay=0.9, center=True, scale=True, epsilon=epsilon,
                                        activation_fn=activation_fn,
                                        param_initializers={'beta': tf.constant_initializer(0.),
                                                            'gamma': tf.random_normal_initializer(1., 0.02)},
                                        reuse=reuse, is_training=is_training, scope=name)


def fc_batch_normalization(inputs, name, epsilon=1e-5, is_training=True, activation_fn=None, reuse=None):
    ori_shape = inputs.get_shape()
    if ori_shape[0] is None:
        ori_shape = -1
    new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
    x = tf.reshape(inputs, new_shape)
    normalized_x = conv_batch_normalization(x, name, epsilon=epsilon, is_training=is_training,
                                            activation_fn=activation_fn, reuse=reuse)
    return tf.reshape(normalized_x, ori_shape)


def reshape(inputs, shape, name):
    return tf.reshape(inputs, shape, name)


def Conv2d(inputs, k_h, k_w, c_o, s_h, s_w, name, activation_fn=None, reuse=None, padding='SAME', biased=False):
    c_i = inputs.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(inputs, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        return output


def Deconv2d(inputs, output_shape, name, k_h, k_w, s_h=2, s_w=2, reuse=None, activation_fn=None, biased=False):
    output_shape[0] = inputs.get_shape()[0]
    ts_output_shape = tf.stack(output_shape)
    w_init = tf.random_normal_initializer(stddev=0.02)

    deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape=ts_output_shape, strides=[1, s_h, s_w, 1])
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, output_shape[-1], inputs.get_shape()[-1]],
                                 initializer=w_init)
        output = deconvolve(inputs, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[output_shape[-1]])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        deconv = tf.reshape(output, [-1] + output_shape[1:])

        return deconv


def add(inputs, name):
    return tf.add_n(inputs, name=name)


def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)


def flatten(inputs, name):
    input_shape = inputs.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        inputs = tf.reshape(inputs, [-1, dim])

    return inputs
