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

import tensorflow as tf

from .layers import conv2d_bn_relu, basic_block


class ResNet(object):
    BASE_FILTERS = 64

    def __init__(self,
                 block,
                 num_blocks_arr,
                 use_bn=True,
                 is_training=False,
                 use_deconv_arr=False,
                 deconv=None):
        self.block = block
        self.num_blocks_arr = num_blocks_arr
        self.is_training = is_training
        self.use_bn = use_bn

        self.use_deconv_arr = [
            use_deconv_arr for _ in range(len(num_blocks_arr) + 1)
        ] if isinstance(use_deconv_arr, bool) else use_deconv_arr
        self.deconv = deconv

    def _stage0(self, inputs, filters):
        outputs = conv2d_bn_relu(inputs,
                                 filters, (7, 7),
                                 2,
                                 'same',
                                 use_bn=self.use_bn,
                                 is_training=self.is_training)
        outputs = tf.layers.max_pooling2d(outputs,
                                          pool_size=3,
                                          strides=2,
                                          padding='same')
        return outputs

    def _make_stage(self, inputs, num_blocks, filters, strides, use_deconv):
        for i in range(num_blocks):
            inputs = self.block(
                inputs,
                filters,
                strides=1 if i else strides,
                conv=self.deconv if use_deconv else tf.layers.conv2d,
                use_bn=self.use_bn,
                is_training=self.is_training)
        return inputs

    def call(self, inputs):
        stages = []
        stages.append(self._stage0(inputs, self.BASE_FILTERS))

        FILTERS = [
            2**x * self.BASE_FILTERS for x in range(len(self.num_blocks_arr))
        ]
        for idx, (num_blocks, filters, use_deconv) in enumerate(
                zip(self.num_blocks_arr, FILTERS, self.use_deconv_arr)):
            stages.append(
                self._make_stage(stages[-1], num_blocks, filters,
                                 2 if idx else 1, use_deconv))

        return stages


def resnet18(inputs, **kwargs):
    return ResNet(basic_block, (2, 2, 2, 2), **kwargs).call(inputs)


def resnet34(inputs, **kwargs):
    return ResNet(basic_block, (3, 4, 6, 3), **kwargs).call(inputs)