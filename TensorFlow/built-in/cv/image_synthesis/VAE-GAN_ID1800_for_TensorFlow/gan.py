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
'''TensorFlow implementation of http://arxiv.org/pdf/1511.06434.pdf'''

from __future__ import absolute_import, division, print_function
from npu_bridge.npu_init import *

import math

import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

from utils import discriminator, decoder
from generator import Generator

def concat_elu(inputs):
    return tf.nn.elu(tf.concat(3, [-inputs, inputs]))

class GAN(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate):
        self.input_tensor = tf.placeholder(tf.float32, [None, 28 * 28])

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=concat_elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True}):
            with tf.variable_scope("model"):
                D1 = discriminator(self.input_tensor)  # positive examples
                D_params_num = len(tf.trainable_variables())
                G = decoder(tf.random_normal([batch_size, hidden_size]))
                self.sampled_tensor = G

            with tf.variable_scope("model", reuse=True):
                D2 = discriminator(G)  # generated examples

        D_loss = self.__get_discrinator_loss(D1, D2)
        G_loss = self.__get_generator_loss(D2)

        params = tf.trainable_variables()
        D_params = params[:D_params_num]
        G_params = params[D_params_num:]
        #    train_discrimator = optimizer.minimize(loss=D_loss, var_list=D_params)
        # train_generator = optimizer.minimize(loss=G_loss, var_list=G_params)
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_discrimator = layers.optimize_loss(
            D_loss, global_step, learning_rate / 10, 'Adam', variables=D_params, update_ops=[])
        self.train_generator = layers.optimize_loss(
            G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])

        self.sess = tf.Session(config=npu_config_proto())
        self.sess.run(tf.global_variables_initializer())

    def __get_discrinator_loss(self, D1, D2):
        '''Loss for the discriminator network

        Args:
            D1: logits computed with a discriminator networks from real images
            D2: logits computed with a discriminator networks from generated images

        Returns:
            Cross entropy loss, positive samples have implicit labels 1, negative 0s
        '''
        return (losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))) +
                losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1))))

    def __get_generator_loss(self, D2):
        '''Loss for the genetor. Maximize probability of generating images that
        discrimator cannot differentiate.

        Returns:
            see the paper
        '''
        return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))

    def update_params(self, inputs):
        d_loss_value = self.sess.run(self.train_discrimator, {
            self.input_tensor: inputs})

        g_loss_value = self.sess.run(self.train_generator)

        return g_loss_value

