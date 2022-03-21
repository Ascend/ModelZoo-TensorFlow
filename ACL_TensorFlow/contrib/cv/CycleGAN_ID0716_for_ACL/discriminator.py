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
import ops


class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, input_data):
        """
    Args:
      input_data: batch_size x image_size x image_size x 3
    Returns:
      output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
              filled with 0.9 if real, 0.0 if fake
    """
        with tf.variable_scope(self.name):
            # convolution layers
            C64 = ops.Ck(input_data, 64, reuse=self.reuse, norm=None,
                         is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
            C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
            C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
            C512 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)

            # apply a convolution to produce a 1 dimensional output (1 channel?)
            # use_sigmoid = False if use_lsgan = True
            output = ops.last_conv(C512, reuse=self.reuse,
                                   use_sigmoid=self.use_sigmoid, name='output')  # (?, w/16, h/16, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
