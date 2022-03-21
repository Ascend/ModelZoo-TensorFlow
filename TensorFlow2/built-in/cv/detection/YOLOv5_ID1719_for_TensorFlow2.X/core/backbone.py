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
import core.common as common

# implement of yolov5s
# TODO implement yolov5 using scale multiplier
def yolov5CSP(x, depth_gain=0.33, width_gain=0.5):
    x = common.Focus(x, c2 = 64 * width_gain)
    x = common.convolutional(x, filters_shape=(3, 3, 64 * width_gain, 128 * width_gain), downsample=True, activate_type='mish')
    x = common.C3(x, 128 * width_gain, 128 * width_gain, 3 * depth_gain)
    x = common.convolutional(x, filters_shape=(3, 3, 128 * width_gain, 256 * width_gain), downsample=True, activate_type='mish')
    x = common.C3(x, 256 * width_gain, 256 * width_gain, 9 * depth_gain)
    route1 = x
    x = common.convolutional(x, filters_shape=(3, 3, 256 * width_gain, 512 * width_gain), downsample=True, activate_type='mish')
    x = common.C3(x, 512 * width_gain, 512 * width_gain, 9 * depth_gain)
    route2 = x
    x = common.convolutional(x, filters_shape=(3, 3, 512 * width_gain, 1024 * width_gain), downsample=True, activate_type='mish')
    x = common.SPP(x, 1024 * width_gain, 1024 * width_gain)
    x = common.C3(x, 1024 * width_gain, 1024 * width_gain, 3 * depth_gain, False)
    return route1, route2, x

def darknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32))
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64)

    input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128,  64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def cspdarknet53(input_data):

    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data

def cspdarknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data

def darknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 256, 512))
    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data


