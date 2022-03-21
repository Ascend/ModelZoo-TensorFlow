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
"""13-Layer Max-Pooling ConvNet with BatchNormalization."""
import npu_bridge.npu_init 

from keras import initializers
from keras.models import Model
from keras.regularizers import l2
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import Input, Conv2D, Dropout, LeakyReLU

leakiness = 0.1
weight_decay = 0.0005
initer = initializers.he_normal()

bn_params = dict(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        gamma_initializer='ones',
    )

conv_params = dict(
        use_bias=True,
        kernel_initializer=initer,
        kernel_regularizer=l2(weight_decay),
    )


def create_network(input_shape, dropout=0.0):
    """
    create_network.
    """
    data = Input(shape=input_shape) 

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(data)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (3, 3), padding='same', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(512, (3, 3), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(256, (1, 1), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    x = Conv2D(128, (1, 1), padding='valid', **conv_params)(x)
    x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(leakiness)(x)

    # Return output dimensions 6 x 6 x 128
    net = Model(data, x, name='convnet_trunk')
    return net


