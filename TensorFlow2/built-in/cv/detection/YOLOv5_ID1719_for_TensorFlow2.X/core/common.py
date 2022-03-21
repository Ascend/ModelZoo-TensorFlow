#! /usr/bin/env python
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

import tensorflow as tf
# import tensorflow_addons as tfa
def Focus(input_layer, c1=12, c2=32, k=3):
    """
    Focus wh information into c-space
    n, c1/4, w, h -> n, c2, w/2, h/2
    :param input_layer: input of size (n, c1, w, h)
    :param c1: in channel (not configurable, shown here for understanding)
    :param c2: out channel
    :param k: kernel size
    :return: output of size (n, c2, w/2, h/2)
    """
    return convolutional(tf.concat([input_layer[:, ::2, ::2, :], input_layer[:, 1::2, ::2, :], input_layer[:, ::2, 1::2, :], input_layer[:, 1::2, 1::2, :]], axis=-1), filters_shape=(k, k, c1, c2), activate_type='mish')

def Bottleneck(input_layer, c1, c2, shortcut=True, e=0.5):
    """
    common bottleneck
    :param input_layer: input features
    :param c1: in channel
    :param c2: out channel
    :param shortcut: add shortcut or not
    :param e: expansion
    :return:
    """
    c_ = int(c2 * e)
    route1 = input_layer
    route2 = convolutional(input_layer, filters_shape=(1, 1, c1, c_), activate_type='mish')
    route2 = convolutional(route2, filters_shape=(3, 3, c_, c2), activate_type='mish')
    add = shortcut and c1 == c2
    return route1 + route2 if add else route2
    
def C3(input_layer, c1, c2, n, shortcut=True, e=0.5):
    """
    CSP Bottleneck with 3 convolutions
    :param input_layer: input features
    :param c1: in channel
    :param c2: out channel
    :param n: number of bottleneck blocks
    :param e: expansion
    :return: output
    """
    c_ = int(c2 * e)
    route1 = convolutional(input_layer, filters_shape=(1, 1, c1, c_), activate_type='mish')
    route2 = convolutional(input_layer, filters_shape=(1, 1, c1, c_), activate_type='mish')
    for i in range(round(n)):
        route1 = Bottleneck(route1, c_, c_, shortcut, e=1)
    fused = tf.concat([route1, route2], axis=-1)
    return convolutional(fused, filters_shape=(1, 1, 2 * c_, c2), activate_type='mish')
    
def SPP(input_layer, c1, c2, k=(5, 9, 13)):
    """
    spatial pyramid pooling layer
    :param input_layer: input features
    :param c1: in channel
    :param c2: out channel
    :param k: pooling kernel size
    :return: output
    """
    c_ = c1 // 2
    route1 = convolutional(input_layer, filters_shape=(1, 1, c1, c_), activate_type='mish')
    routes = [route1]
    for x in k:
        route = tf.nn.max_pool(route1, x, 1, 'SAME')
        routes.append(route)
    fused = tf.concat(routes, axis=-1)
    return convolutional(fused, filters_shape=(1, 1, c_ * (len(k)+1), c2),activate_type='mish')

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

