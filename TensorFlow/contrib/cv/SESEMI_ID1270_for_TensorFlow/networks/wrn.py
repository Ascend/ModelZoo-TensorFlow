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
"""Wide Residual Networks with Max Pooling.
Reference - https://arxiv.org/abs/1605.07146
"""
import npu_bridge.npu_init

from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers import BatchNormalization, Add

leakiness = 0.0
weight_decay = 0.0005
initer = initializers.he_normal()

bn_params = dict(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        gamma_initializer='ones',
    )

conv_params = dict(
        use_bias=False,
        kernel_initializer=initer,
        kernel_regularizer=l2(weight_decay),
    )


def create_network(input_shape, dropout=0.0):
    """
    For WRN depth 16: set N = (16 - 4) / 6 = 2
    For WRN depth 28: set N = (28 - 4) / 6 = 4
    For WRN depth 40: set N = (40 - 4) / 6 = 6
    """
    depth = 28
    width = 2
    
    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that'
                         '(depth - 4) is divisible by 6.')

    N = (depth - 4) // 6

    data = Input(shape=input_shape)
    
    x = initial_conv(data)
    nb_conv = 4

    x = expand_conv(x, 16, width)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, width, dropout)
        nb_conv += 2

    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = expand_conv(x, 32, width, strides=(1, 1))
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, width, dropout)
        nb_conv += 2

    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = expand_conv(x, 64, width, strides=(1, 1))
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, width, dropout)
        nb_conv += 2

    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    # Return output dimensions 8 x 8 x 128
    net = Model(data, x, name='wrn_trunk')
    print("Wide Residual Network-%d-%d." % (nb_conv, width))
    return net


def initial_conv(input):
    """
    initialize one converlution layers.
    return the layer.
    """
    x = Conv2D(16, (3, 3), padding='same', **conv_params)(input)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)
    return x


def expand_conv(init, base, k, stride=None):
    """
    expand.
    init_x.
    return.
    """
    if stride is None:
           stride=(1, 1)
    x = Conv2D(base * k, (3, 3), padding='same',
               strides=strides, **conv_params)(init)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(base * k, (3, 3), padding='same', **conv_params)(x)

    skip = Conv2D(base * k, (1, 1), padding='same',
                  strides=strides, **conv_params)(init)

    m = Add()([x, skip])
    return m


def conv1_block(input, k=1, dropout=0.0):
    """
    conv1.
    return.
    """
    init = input
    
    x = BatchNormalization(**bn_params)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(16 * k, (3, 3), padding='same', **conv_params)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)
    
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(16 * k, (3, 3), padding='same', **conv_params)(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0):
    """
    conv2.
    return.
    """
    init = input
    
    x = BatchNormalization(**bn_params)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(32 * k, (3, 3), padding='same', **conv_params)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(32 * k, (3, 3), padding='same', **conv_params)(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0):
    """
    conv3.
    return.
    """
    init = input

    x = BatchNormalization(**bn_params)(input)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(64 * k, (3, 3), padding='same', **conv_params)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)
    x = Conv2D(64 * k, (3, 3), padding='same', **conv_params)(x)

    m = Add()([init, x])
    return m


