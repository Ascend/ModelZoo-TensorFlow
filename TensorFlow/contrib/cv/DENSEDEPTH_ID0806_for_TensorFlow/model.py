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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Concatenate, UpSampling2D
import os
import tensorflow as tf
import sys


def create_model(is_twohundred=False, is_halffeatures=True):
    # with compat.forward_compatibility_horizon(2019, 5, 1):
    # print('Loading base model (DenseNet)..')

    code_dir = os.path.dirname(__file__)
    print("---------code_dir--------------",code_dir)
    # Encoder Layers
    if is_twohundred:
        print("Load densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5")
        if os.path.isfile(os.path.join(code_dir, 'dataset/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')):
            base_model = tf.keras.applications.DenseNet201(input_shape=(480, 640, 3), include_top=False, weights=None)
            base_model.load_weights(
                os.path.join(code_dir, 'dataset/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'))
        else:
            try:
                base_model = tf.keras.applications.DenseNet201(input_shape=(480, 640, 3), include_top=False)
            except Exception:
                print(
                    "Please download the file densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5 and set it into dataset/")
                sys.exit()
    else:

        # input_tensor = tf.keras.Input(shape=(480, 640, 3), batch_size=batch_size, name="input",
        # dtype="float32")
        # base_model = tf.keras.applications.DenseNet169(input_tensor=input_tensor,
        # include_top=False)

        # base_model = tf.keras.applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)

        print("Load densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5")
        if os.path.isfile(os.path.join(code_dir, 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')):
            base_model = tf.keras.applications.DenseNet169(input_shape=(480, 640, 3), include_top=False, weights=None)
            base_model.load_weights(
                os.path.join(code_dir, 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'))
        else:
            try:
                base_model = tf.keras.applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)
            except Exception:
                print(
                    "Please download the file densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5 and set it into dataset/")
                sys.exit()

    # print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True

    # Starting number of decoder filters
    if is_halffeatures:
        decode_filters = int(int(base_model_output_shape[-1]) / 2)
    else:
        decode_filters = int(base_model_output_shape[-1])

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):

        base_model_concat_with_output = base_model.get_layer(concat_with).output

        # base_model_concat_with_output_shape = base_model.get_layer(concat_with).output.shape

        # up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
        #
        # up_i = Reshape((base_model_concat_with_output_shape[1],
        #                 base_model_concat_with_output_shape[2],
        #                 up_i.shape[-1]))(up_i)

        up_i = UpSampling2D(size=(2, 2), interpolation="bilinear", name=name + '_upsampling2d')(tensor)

        up_i = Concatenate(name=name + '_concat')(
            [up_i, base_model_concat_with_output])  # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                     name='conv2')(base_model.output)

    decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1/relu')
    if False: decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')

    # Extract depths (final layer)
    conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

    # Create the model
    model = Model(inputs=base_model.input, outputs=conv3)

    return model
