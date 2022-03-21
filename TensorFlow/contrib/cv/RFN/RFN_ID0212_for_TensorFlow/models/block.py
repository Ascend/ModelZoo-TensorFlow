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
import tensorflow.nn as nn


def norm(
    inputs,
    decay=0.999,
    epsilon=0.001,
    scale=False,
    center=True,
    is_training=True,
    data_format='NHWC',
    param_initializers=None
):
    """Adds a Batch Normalization layer."""

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError(
            "Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    input_shape = inputs.get_shape()
    input_rank = input_shape.ndims
    input_channels = input_shape[1]

    if input_rank == 2:

        if data_format == 'NCHW':
            new_shape = [-1, input_channels, 1, 1]
        else:
            new_shape = [-1, 1, 1, input_channels]

        inputs = tf.reshape(inputs, new_shape)

    output = tf.contrib.layers.batch_norm(
        inputs,
        decay=decay,
        scale=scale,
        epsilon=epsilon,
        is_training=is_training,
        trainable=is_training,
        fused=True,
        data_format=data_format,
        center=center,
        param_initializers=param_initializers
    )

    if input_rank == 2:
        output = tf.reshape(output, [-1, input_channels])

    return output


def get_valid_padding(kernel_size, dilation):
    """Calculate the padding number of valid pattern"""

    new_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    paddings = (new_size - 1) // 2
    if dilation > 1:
        print('kernel_size:', kernel_size, 'dilation:', dilation,
              'new_size:', new_size, 'paddings:', paddings)
    return paddings - 1


def pad(inputs, paddings, mode='CONSTANT', name='padding', constant_values=0):
    """Architecture of padding layer"""

    if mode.upper() not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
        raise ValueError(
            "Unknown padding mode: `%s` (accepted: ['CONSTANT', 'REFLECT', 'SYMMETRIC'])" % mode)

    output = tf.pad(inputs, paddings=[[0, 0], [paddings, paddings], [paddings, paddings], [
                    0, 0]], mode=mode, name=name, constant_values=constant_values)

    return output


def conv_layer(
    inputs,
    filters=8,
    kernel_size=3,
    strides=1,
    dilation_rate=1,
    paddings=0,
    data_format='NHWC',
    use_bias=True,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer(),
    trainable=True
):
    """Architecture of convolutional layer"""

    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError(
            "Unknown data format: `%s` (accepted: ['NHWC', 'NCHW'])" % data_format)

    paddings = ((kernel_size - 1) // 2) * dilation_rate
    output = pad(inputs, paddings)

    output = tf.layers.conv2d(
        output,
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(strides, strides),
        padding="valid",
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        activation=None
    )

    return output


def activation(inputs, act_type, alpha=0.2):
    """Activation layer"""
    act_type = act_type.lower()
    if act_type == 'relu':
        output = tf.nn.relu(inputs)
    elif act_type == 'lrelu':
        output = tf.nn.leaky_relu(inputs, alpha=alpha)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return output


def conv_block(inputs, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    """Architecture of conv block"""

    paddings = get_valid_padding(kernel_size, dilation)
    net = conv_layer(inputs, filters=out_nc, kernel_size=kernel_size,
                     strides=stride, paddings=paddings, dilation_rate=dilation, use_bias=bias)
    net = activation(net, act_type) if act_type else net
    net = norm(net) if norm_type else net
    return net


def _ResBlock_32(inputs, nc=64):
    """Architecture of HFFB"""

    c1 = conv_layer(inputs, filters=nc, kernel_size=3,
                    strides=1, dilation_rate=1)
    output1 = activation(c1, act_type="lrelu")
    d1 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=1)
    d2 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=2)
    d3 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=3)
    d4 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=4)
    d5 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=5)
    d6 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=6)
    d7 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=7)
    d8 = conv_layer(output1, filters=nc//2, kernel_size=3,
                    strides=1, dilation_rate=8)

    add1 = d1 + d2
    add2 = add1 + d3
    add3 = add2 + d4
    add4 = add3 + d5
    add5 = add4 + d6
    add6 = add5 + d7
    add7 = add6 + d8

    combine = tf.concat(
        values=[d1, add1, add2, add3, add4, add5, add6, add7], axis=3)
    c2 = activation(combine, act_type="lrelu")
    output2 = conv_layer(c2, filters=nc, kernel_size=1,
                         strides=1, dilation_rate=1)
    output = inputs + output2 * 0.2

    return output


def RRBlock_32(inputs):
    """Architecture of RRFB"""
    out = _ResBlock_32(inputs)
    out = _ResBlock_32(out)
    out = _ResBlock_32(out)

    return inputs + out * 0.2


def upconv_block(inputs, out_channels, upscale_factor=2, kernel_size=3, stride=1, act_type='relu'):
    """Architecture of upconv block"""
    output = tf.keras.layers.UpSampling2D(
        size=(upscale_factor, upscale_factor), interpolation='nearest')(inputs)
    output = conv_layer(output, out_channels,
                        kernel_size=kernel_size, strides=stride)
    output = activation(output, act_type=act_type)

    return output


def pixelshuffle_block(inputs, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    """Architecture of upsample layer"""
    output = conv_layer(inputs, n_channels=out_channels *
                        (upscale_factor ** 2), kernel_size=kernel_size, strides=stride)
    output = tf.nn.depth_to_space(output, block_size=upscale_factor)

    return output
