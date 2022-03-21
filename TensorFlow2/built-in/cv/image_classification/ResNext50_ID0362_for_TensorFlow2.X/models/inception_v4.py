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
from models.inception_modules import Stem, InceptionBlockA, InceptionBlockB, \
    InceptionBlockC, ReductionA, ReductionB
from configuration import NUM_CLASSES


def build_inception_block_a(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockA())
    return block


def build_inception_block_b(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockB())
    return block


def build_inception_block_c(n):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionBlockC())
    return block


class InceptionV4(tf.keras.Model):
    def __init__(self):
        super(InceptionV4, self).__init__()
        self.stem = Stem()
        self.inception_a = build_inception_block_a(4)
        self.reduction_a = ReductionA(k=192, l=224, m=256, n=384)
        self.inception_b = build_inception_block_b(7)
        self.reduction_b = ReductionB()
        self.inception_c = build_inception_block_c(3)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8))
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=True, mask=None):
        x = self.stem(inputs, training=training)
        x = self.inception_a(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.inception_b(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.inception_c(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        x = self.fc(x)

        return x
