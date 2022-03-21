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
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import batch_norm
from net.layers import *
from net.utils import *


class ClsHead():

    def __init__(self, base_channel, num_branches, cls_num, fc_channel):
        '''

        :param base_channel: C, 2C, 4C, 8C
        :param cls_num: classification number, 1000 for imagenet
        :param fc_channel: Full connected layer's input feature
        '''

        self.base_channel = base_channel
        self.num_branches = num_branches
        self.final_layer_num = 2048
        self.cls_num = cls_num
        self.fc_channels = fc_channel
        self.downsamplefn = downsample_block
        self.bottleneckfn = bottleneck_block
        self.scope = 'CLS_HEAD'

    def forward(self, inputs):
        assert len(inputs) == self.num_branches, \
            "input_channel {} must to be same as num_branches {}".format(len(inputs), self.num_branches)

        num_channels = [self.base_channel * (2 ** i) for i in range(self.num_branches)]

        # make trans layers to [1C, 2C, 4C, 8C]
        trans_features = self.__make_trans_layers(inputs, num_channels)

        # make downsample and add layers
        for i in range(self.num_branches):
            if i < self.num_branches - 1:
                scope = self.scope + '_DS' + str(i)
                out = self.__downsample_and_add(trans_features[i], trans_features[i + 1], num_channels[i + 1], scope)
                trans_features[i + 1] = out

        # final layer
        final_output = conv1x1_block(out, self.final_layer_num, scope=self.scope + '_final', has_relu=True)

        with tf.variable_scope(self.scope):
            # average pooling
            output = tf.reduce_mean(final_output, [1, 2], keepdims=True)
            output = tf.squeeze(output, [1, 2])

            # fc
            output = tf.layers.dense(inputs=output, units=self.cls_num)
            output = tf.identity(output, 'final_dense')

        return output

    def __make_trans_layers(self, inputs, num_channels):
        outputs = []
        for i, _input in enumerate(inputs):
            _output = bottleneck_block(_input, num_channels[i], scope=self.scope + '_BK' + str(i),
                                       downsamplefn=trans_block)
            outputs.append(_output)
        return outputs

    def __downsample_and_add(self, fa, fb, out_channel, scope):

        dw_fa = self.downsamplefn(fa, out_channel, scope, has_relu=True)

        return dw_fa + fb
