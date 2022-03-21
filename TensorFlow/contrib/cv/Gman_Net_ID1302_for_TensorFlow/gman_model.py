"""
model
"""
# coding=utf-8
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

import gman_tools as tools
import gman_flags as flags


# This model consists with the one metioned in paper
# Generic Model-Agnostic Convolutional Neural Network for Single Image Dehazing
# https://arxiv.org/ptools/1810.02862.ptools
class Gman(object):
    """
    v1
    """

    @staticmethod
    def inference(input_data, batch_size=None, h=None, w=None):
        """
        The forward process of network.
        :param input_data:  Batch used to for training, always in size of [batch_size, h, w, 3]
        :param batch_size:  1 for evaluation and custom number for training.
        :param h: height of the image
        :param w: width of the image
        :return: The result processed by gman
        """
        if h is None or w is None or batch_size is None:
            h = flags.FLAGS.input_image_height
            w = flags.FLAGS.input_image_width
            batch_size = flags.FLAGS.batch_size
        with tf.variable_scope('DehazeNet'):
            x_s = input_data
            # ####################################################################
            # #####################Two convolutional layers###########################
            # ####################################################################
            x = tools.conv('DN_conv1_1', x_s, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            # ####################################################################
            # ###################Two Downsampling layers############################
            # ####################################################################
            x = tools.conv('upsampling_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.conv('upsampling_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

            # ####################################################################
            # #######################Residual Blocks#################################
            # ####################################################################
            x1 = tools.conv('DN_conv2_1', x, 128, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv2_2', x1, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv2_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x1)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x2 = tools.conv('DN_conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv3_2', x2, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x2)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x3 = tools.conv('DN_conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_2', x3, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv4_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x3)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x4 = tools.conv('DN_conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_2', x4, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv5_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x4)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # ####################################################################
            # #####################Two deconvolutional layers#########################
            # ####################################################################
            x = tools.deconv('DN_deconv1', x, 64, 64, output_shape=[batch_size, int((h + 1) / 2), int((w + 1) / 2), 64],
                             kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.deconv('DN_deconv2', x, 64, 64, output_shape=[batch_size, h, w, 64], kernel_size=[3, 3],
                             stride=[1, 2, 2, 1])

            x_r = tools.conv_nonacti('DN_conv7_1', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x_r = tf.add(x_r, x_s)
            x_r = tools.acti_layer(x_r)

            return x_r


if __name__ == '__main__':
    pass
