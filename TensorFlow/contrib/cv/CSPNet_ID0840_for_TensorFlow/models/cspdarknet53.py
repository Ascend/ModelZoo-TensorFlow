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
# Copyright 2022 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *
import tensorflow as tf
import utils as tu


def Conv2dBatchLeaky(x, kernel_size, in_channels, out_channels, conv_stride, stage_name):
    w = tu.weight([kernel_size, kernel_size, in_channels, out_channels], name='w_'+stage_name)
    b = tu.bias(0.0, [out_channels], name='b_'+stage_name)
    x = tf.add(tu.conv2d(x, w, stride=(conv_stride, conv_stride), padding='SAME'), b)
    x = tu.batch_norm(x)
    x = x * tf.math.tanh(tf.math.softplus(x))
    return x


def SmallBlock(x, channels, stage_name):
    c1 = Conv2dBatchLeaky(x, 1, channels, channels, 1, stage_name+'c1')
    c2 = Conv2dBatchLeaky(c1, 3, channels, channels, 1, stage_name+'c2')
    return c2


def classifier(x):
    with tf.name_scope('stage1') as scope:
        with tf.name_scope('stage1_Conv2dBatchLeaky') as inner_scope:
            s1c1 = Conv2dBatchLeaky(x, 3, 3, 32, 1, 's1c1')

    with tf.name_scope('stage2') as scope:
        with tf.name_scope('stage2_Conv2dBatchLeaky_1') as inner_scope:
            s2c1 = Conv2dBatchLeaky(s1c1, 3, 32, 64, 2, 's2c1')
        with tf.name_scope('stage2_Split_0') as inner_scope:
            s2s0 = Conv2dBatchLeaky(s2c1, 1, 64, 64, 1, 's2s0')
        with tf.name_scope('stage2_Split_1') as inner_scope:
            s2s1 = Conv2dBatchLeaky(s2c1, 1, 64, 64, 1, 's2s1')
        with tf.name_scope('stage2_Conv2dBatchLeaky_2') as inner_scope:
            s2c2 = Conv2dBatchLeaky(s2s1, 1, 64, 32, 1, 's2c2')
        with tf.name_scope('stage2_Conv2dBatchLeaky_3') as inner_scope:
            s2c3 = Conv2dBatchLeaky(s2c2, 3, 32, 64, 1, 's2c3')
        shortcut2 = tf.add(s2c3, s2s1)
        with tf.name_scope('stage2_Conv2dBatchLeaky_4') as inner_scope:
            s2c4 = Conv2dBatchLeaky(shortcut2, 1, 64, 64, 1, 's2c4')
        route2 = tf.concat([s2s0, s2c4], axis=3)

    with tf.name_scope('stage3') as scope:
        with tf.name_scope('stage3_Conv2dBatchLeaky_1') as inner_scope:
            s3c1 = Conv2dBatchLeaky(route2, 1, 128, 64, 1, 's3c1')
        with tf.name_scope('stage3_Conv2dBatchLeaky_2') as inner_scope:
            s3c2 = Conv2dBatchLeaky(s3c1, 3, 64, 128, 2, 's3c2')
        with tf.name_scope('stage3_Split_0') as inner_scope:
            s3s0 = Conv2dBatchLeaky(s3c2, 1, 128, 64, 1, 's3s0')
        with tf.name_scope('stage3_Split_1') as inner_scope:
            s3s1 = Conv2dBatchLeaky(s3c2, 1, 128, 64, 1, 's3s1')
        with tf.name_scope('stage3_Block_1') as inner_scope:
            s3b1 = SmallBlock(s3s1, 64, 's3b1')
        with tf.name_scope('stage3_Block_2') as inner_scope:
            s3b2 = SmallBlock(s3b1, 64, 's3b2')
        with tf.name_scope('stage3_Conv2dBatchLeaky_3') as inner_scope:
            s3c3 = Conv2dBatchLeaky(s3b2, 1, 64, 64, 1, 's3c3')
        route3 = tf.concat([s3s0, s3c3], axis=3)

    with tf.name_scope('stage4') as scope:
        with tf.name_scope('stage4_Conv2dBatchLeaky_1') as inner_scope:
            s4c1 = Conv2dBatchLeaky(route3, 1, 128, 128, 1, 's4c1')
        with tf.name_scope('stage4_Conv2dBatchLeaky_2') as inner_scope:
            s4c2 = Conv2dBatchLeaky(s4c1, 3, 128, 256, 2, 's4c2')
        with tf.name_scope('stage4_Split_0') as inner_scope:
            s4s0 = Conv2dBatchLeaky(s4c2, 1, 256, 128, 1, 's4s0')
        with tf.name_scope('stage4_Split_1') as inner_scope:
            s4s1 = Conv2dBatchLeaky(s4c2, 1, 256, 128, 1, 's4s1')

        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b1 = SmallBlock(s4s1, 128, 's4b1')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b2 = SmallBlock(s4b1, 128, 's4b2')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b3 = SmallBlock(s4b2, 128, 's4b3')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b4 = SmallBlock(s4b3, 128, 's4b4')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b5 = SmallBlock(s4b4, 128, 's4b5')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b6 = SmallBlock(s4b5, 128, 's4b6')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b7 = SmallBlock(s4b6, 128, 's4b7')
        with tf.name_scope('stage4_Block_1') as inner_scope:
            s4b8 = SmallBlock(s4b7, 128, 's4b8')

        with tf.name_scope('stage4_Conv2dBatchLeaky_3') as inner_scope:
            s4c3 = Conv2dBatchLeaky(s4b8, 1, 128, 128, 1, 's4c3')
        route4 = tf.concat([s4s0, s4c3], axis=3)

    with tf.name_scope('stage5') as scope:
        with tf.name_scope('stage5_Conv2dBatchLeaky_1') as inner_scope:
            s5c1 = Conv2dBatchLeaky(route4, 1, 256, 256, 1, 's5c1')
        with tf.name_scope('stage5_Conv2dBatchLeaky_2') as inner_scope:
            s5c2 = Conv2dBatchLeaky(s5c1, 3, 256, 512, 2, 's5c2')
        with tf.name_scope('stage5_Split_0') as inner_scope:
            s5s0 = Conv2dBatchLeaky(s5c2, 1, 512, 256, 1, 's5s0')
        with tf.name_scope('stage5_Split_1') as inner_scope:
            s5s1 = Conv2dBatchLeaky(s5c2, 1, 512, 256, 1, 's5s1')

        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b1 = SmallBlock(s5s1, 256, 's5b1')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b2 = SmallBlock(s5b1, 256, 's5b2')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b3 = SmallBlock(s5b2, 256, 's5b3')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b4 = SmallBlock(s5b3, 256, 's5b4')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b5 = SmallBlock(s5b4, 256, 's5b5')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b6 = SmallBlock(s5b5, 256, 's5b6')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b7 = SmallBlock(s5b6, 256, 's5b7')
        with tf.name_scope('stage5_Block_1') as inner_scope:
            s5b8 = SmallBlock(s5b7, 256, 's5b8')

        with tf.name_scope('stage5_Conv2dBatchLeaky_3') as inner_scope:
            s5c3 = Conv2dBatchLeaky(s5b8, 1, 256, 256, 1, 's5c3')
        route5 = tf.concat([s5s0, s5c3], axis=3)

    with tf.name_scope('stage6') as scope:
        with tf.name_scope('stage6_Conv2dBatchLeaky_1') as inner_scope:
            s6c1 = Conv2dBatchLeaky(route5, 1, 512, 512, 1, 's6c1')
        with tf.name_scope('stage6_Conv2dBatchLeaky_2') as inner_scope:
            s6c2 = Conv2dBatchLeaky(s6c1, 3, 512, 1024, 2, 's6c2')
        with tf.name_scope('stage6_Split_0') as inner_scope:
            s6s0 = Conv2dBatchLeaky(s6c2, 1, 1024, 512, 1, 's6s0')
        with tf.name_scope('stage6_Split_1') as inner_scope:
            s6s1 = Conv2dBatchLeaky(s6c2, 1, 1024, 512, 1, 's6s1')

        with tf.name_scope('stage6_Block_1') as inner_scope:
            s6b1 = SmallBlock(s6s1, 512, 's6b1')
        with tf.name_scope('stage6_Block_1') as inner_scope:
            s6b2 = SmallBlock(s6b1, 512, 's6b2')
        with tf.name_scope('stage6_Block_1') as inner_scope:
            s6b3 = SmallBlock(s6b2, 512, 's6b3')
        with tf.name_scope('stage6_Block_1') as inner_scope:
            s6b4 = SmallBlock(s6b3, 512, 's6b4')


        with tf.name_scope('stage6_Conv2dBatchLeaky_3') as inner_scope:
            s6c3 = Conv2dBatchLeaky(s6b4, 1, 512, 512, 1, 's6c3')
        route6 = tf.concat([s6s0, s6c3], axis=3)

        conv = Conv2dBatchLeaky(route6, 1, 1024, 1024, 1, 'conv')
        avg_pool = tf.squeeze(conv)

        wfc = tu.weight([1024, 10], name='wfc')
        bfc = tu.bias(0.0, [10], name='bfc')
        out = tf.add(tf.matmul(avg_pool, wfc), bfc)

    return out

