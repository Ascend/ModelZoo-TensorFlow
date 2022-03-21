#
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
#

import tensorflow as tf
from configuration import NUM_CLASSES


class MobileNetV1(tf.keras.Model):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.separable_conv_1 = tf.keras.layers.SeparableConv2D(filters=64,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_2 = tf.keras.layers.SeparableConv2D(filters=128,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_3 = tf.keras.layers.SeparableConv2D(filters=128,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_4 = tf.keras.layers.SeparableConv2D(filters=256,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")
        self.separable_conv_5 = tf.keras.layers.SeparableConv2D(filters=256,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_6 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=2,
                                                                padding="same")

        self.separable_conv_7 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_8 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_9 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                kernel_size=(3, 3),
                                                                strides=1,
                                                                padding="same")
        self.separable_conv_10 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                 kernel_size=(3, 3),
                                                                 strides=1,
                                                                 padding="same")
        self.separable_conv_11 = tf.keras.layers.SeparableConv2D(filters=512,
                                                                 kernel_size=(3, 3),
                                                                 strides=1,
                                                                 padding="same")

        self.separable_conv_12 = tf.keras.layers.SeparableConv2D(filters=1024,
                                                                 kernel_size=(3, 3),
                                                                 strides=2,
                                                                 padding="same")
        self.separable_conv_13 = tf.keras.layers.SeparableConv2D(filters=1024,
                                                                 kernel_size=(3, 3),
                                                                 strides=1,
                                                                 padding="same")

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                                         strides=1)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.separable_conv_1(x)
        x = self.separable_conv_2(x)
        x = self.separable_conv_3(x)
        x = self.separable_conv_4(x)
        x = self.separable_conv_5(x)
        x = self.separable_conv_6(x)
        x = self.separable_conv_7(x)
        x = self.separable_conv_8(x)
        x = self.separable_conv_9(x)
        x = self.separable_conv_10(x)
        x = self.separable_conv_11(x)
        x = self.separable_conv_12(x)
        x = self.separable_conv_13(x)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x
    
