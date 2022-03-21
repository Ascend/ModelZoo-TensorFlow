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


class HRFront():

    def __init__(self, num_channels, bottlenect_channels, output_channels, num_blocks):
        self.scope = 'HR_FRONT'
        self.num_channels = num_channels
        self.bottleneck_channels = bottlenect_channels
        self.output_channels = output_channels
        self.num_blocks = num_blocks

    def forward(self, input):
        with tf.variable_scope(self.scope):
            # conv1 + bn1 + relu1
            _out = slim.conv2d(input, num_outputs=self.num_channels, kernel_size=[3, 3],
                               stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            # conv2 + bn2 + relu2
            _out = slim.conv2d(_out, num_outputs=self.num_channels, kernel_size=[3, 3],
                               stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            # bottlenect
            for i in range(self.num_blocks):
                _out = bottleneck_block(_out, planes=self.bottleneck_channels,
                                        scope='_BN' + str(i), downsamplefn=trans_block if i == 0 else None)

            # one 3x3 keep same resolution and one 3x3 to 1/2x resolution
            _same_res = slim.conv2d(_out, num_outputs=self.output_channels[0], kernel_size=[3, 3],
                                    stride=1, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

            _half_res = slim.conv2d(_out, num_outputs=self.output_channels[1], kernel_size=[3, 3],
                                    stride=2, activation_fn=tf.nn.relu, normalizer_fn=batch_norm)

        return [_same_res, _half_res]
