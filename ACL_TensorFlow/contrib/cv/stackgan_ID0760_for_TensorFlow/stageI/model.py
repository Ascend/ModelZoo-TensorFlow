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
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys

sys.path.append('misc')

from custom_ops import fc, conv_batch_normalization, fc_batch_normalization, reshape, Conv2d, Deconv2d, UpSample, add
from config import cfg


class CondGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 = int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

    # g-net
    def generate_condition(self, c_var):
        conditions = fc(c_var, self.ef_dim * 2, 'gen_cond/fc', activation_fn=tf.nn.leaky_relu)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    def generator(self, z_var, training=True):
        node1_0 = fc(z_var, self.s16 * self.s16 * self.gf_dim * 8, 'g_n1.0/fc')
        node1_0 = fc_batch_normalization(node1_0, 'g_n1.0/batch_norm')
        node1_0 = reshape(node1_0, [-1, self.s16, self.s16, self.gf_dim * 8], name='g_n1.0/reshape')

        node1_1 = Conv2d(node1_0, 1, 1, self.gf_dim * 2, 1, 1, name='g_n1.1/conv2d')
        node1_1 = conv_batch_normalization(node1_1, 'g_n1.1/batch_norm_1', activation_fn=tf.nn.relu,
                                           is_training=training)
        node1_1 = Conv2d(node1_1, 3, 3, self.gf_dim * 2, 1, 1, name='g_n1.1/conv2d2')
        node1_1 = conv_batch_normalization(node1_1, 'g_n1.1/batch_norm_2', activation_fn=tf.nn.relu,
                                           is_training=training)
        node1_1 = Conv2d(node1_1, 3, 3, self.gf_dim * 8, 1, 1, name='g_n1.1/conv2d3')
        node1_1 = conv_batch_normalization(node1_1, 'g_n1.1/batch_norm_3', activation_fn=tf.nn.relu,
                                           is_training=training)

        node1 = add([node1_0, node1_1], name='g_n1_res/add')
        node1_output = tf.nn.relu(node1)

        node2_0 = UpSample(node1_output, size=[self.s8, self.s8], method=1, align_corners=False, name='g_n2.0/upsample')
        node2_0 = Conv2d(node2_0, 3, 3, self.gf_dim * 4, 1, 1, name='g_n2.0/conv2d')
        node2_0 = conv_batch_normalization(node2_0, 'g_n2.0/batch_norm', is_training=training)

        node2_1 = Conv2d(node2_0, 1, 1, self.gf_dim * 1, 1, 1, name='g_n2.1/conv2d')
        node2_1 = conv_batch_normalization(node2_1, 'g_n2.1/batch_norm', activation_fn=tf.nn.relu, is_training=training)
        node2_1 = Conv2d(node2_1, 3, 3, self.gf_dim * 1, 1, 1, name='g_n2.1/conv2d2')
        node2_1 = conv_batch_normalization(node2_1, 'g_n2.1/batch_norm2', activation_fn=tf.nn.relu,
                                           is_training=training)
        node2_1 = Conv2d(node2_1, 3, 3, self.gf_dim * 4, 1, 1, name='g_n2.1/conv2d3')
        node2_1 = conv_batch_normalization(node2_1, 'g_n2.1/batch_norm3', is_training=training)

        node2 = add([node2_0, node2_1], name='g_n2_res/add')
        node2_output = tf.nn.relu(node2)

        output_tensor = UpSample(node2_output, size=[self.s4, self.s4], method=1, align_corners=False,
                                 name='g_OT/upsample')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim * 2, 1, 1, name='g_OT/conv2d')
        output_tensor = conv_batch_normalization(output_tensor, 'g_OT/batch_norm', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = UpSample(output_tensor, size=[self.s2, self.s2], method=1, align_corners=False,
                                 name='g_OT/upsample2')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim, 1, 1, name='g_OT/conv2d2')
        output_tensor = conv_batch_normalization(output_tensor, 'g_OT/batch_norm2', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = UpSample(output_tensor, size=[self.s, self.s], method=1, align_corners=False,
                                 name='g_OT/upsample3')
        output_tensor = Conv2d(output_tensor, 3, 3, 3, 1, 1, activation_fn=tf.nn.tanh, name='g_OT/conv2d3')
        return output_tensor

    def generator_simple(self, z_var, training=True):
        output_tensor = fc(z_var, self.s16 * self.s16 * self.gf_dim * 8, 'g_simple_OT/fc')
        output_tensor = reshape(output_tensor, [-1, self.s16, self.s16, self.gf_dim * 8], name='g_simple_OT/reshape')
        output_tensor = conv_batch_normalization(output_tensor, 'g_simple_OT/batch_norm', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = Deconv2d(output_tensor, [0, self.s8, self.s8, self.gf_dim * 4], name='g_simple_OT/deconv2d',
                                 k_h=4, k_w=4)
        output_tensor = conv_batch_normalization(output_tensor, 'g_simple_OT/batch_norm2', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = Deconv2d(output_tensor, [0, self.s4, self.s4, self.gf_dim * 2], name='g_simple_OT/deconv2d2',
                                 k_h=4, k_w=4)
        output_tensor = conv_batch_normalization(output_tensor, 'g_simple_OT/batch_norm3', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = Deconv2d(output_tensor, [0, self.s2, self.s2, self.gf_dim], name='g_simple_OT/deconv2d3',
                                 k_h=4, k_w=4)
        output_tensor = conv_batch_normalization(output_tensor, 'g_simple_OT/batch_norm4', activation_fn=tf.nn.relu,
                                                 is_training=training)
        output_tensor = Deconv2d(output_tensor, [0] + list(self.image_shape), name='g_simple_OT/deconv2d4',
                                 k_h=4, k_w=4, activation_fn=tf.nn.tanh)

        return output_tensor


    def get_generator(self, z_var, is_training):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var, training=is_training)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var, training=is_training)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self, inputs=None, if_reuse=None):
        template = fc(inputs, self.ef_dim, 'd_embedd/fc', activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        return template

    def d_encode_image(self, training=True, inputs=None, if_reuse=None):
        node1_0 = Conv2d(inputs, 4, 4, self.df_dim, 2, 2, name='d_n1.0/conv2d', activation_fn=tf.nn.leaky_relu,
                         reuse=if_reuse)
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 2, 2, 2, name='d_n1.0/conv2d2', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'd_n1.0/batch_norm', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 4, 2, 2, name='d_n1.0/conv2d3', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'd_n1.0/batch_norm2', is_training=training, reuse=if_reuse)
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 8, 2, 2, name='d_n1.0/conv2d4', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'd_n1.0/batch_norm3', is_training=training, reuse=if_reuse)

        node1_1 = Conv2d(node1_0, 1, 1, self.df_dim * 2, 1, 1, name='d_n1.1/conv2d', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'd_n1.1/batch_norm', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        node1_1 = Conv2d(node1_1, 3, 3, self.df_dim * 2, 1, 1, name='d_n1.1/conv2d2', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'd_n1.1/batch_norm2', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        node1_1 = Conv2d(node1_1, 3, 3, self.df_dim * 8, 1, 1, name='d_n1.1/conv2d3', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'd_n1.1/batch_norm3', is_training=training, reuse=if_reuse)

        node1 = add([node1_0, node1_1], name='d_n1_res/add')
        node1 = tf.nn.leaky_relu(node1)

        return node1

    def d_encode_image_simple(self, training=True, inputs=None, if_reuse=None):
        template = Conv2d(inputs, 4, 4, self.df_dim, 2, 2, activation_fn=tf.nn.leaky_relu, name='d_template/conv2d',
                          reuse=if_reuse)
        template = Conv2d(template, 4, 4, self.df_dim * 2, 2, 2, name='d_template/conv2d2', reuse=if_reuse)
        template = conv_batch_normalization(template, 'd_template/batch_norm', is_training=training,
                                            activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        template = Conv2d(template, 4, 4, self.df_dim * 4, 2, 2, name='d_template/conv2d3', reuse=if_reuse)
        template = conv_batch_normalization(template, 'd_template/batch_norm2', is_training=training,
                                            activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        template = Conv2d(template, 4, 4, self.df_dim * 8, 2, 2, name='d_template/conv2d4', reuse=if_reuse)
        template = conv_batch_normalization(template, 'd_template/batch_norm3', is_training=training,
                                            activation_fn=tf.nn.leaky_relu, reuse=if_reuse)

        return template

    def discriminator(self, training=True, inputs=None, if_reuse=None):
        template = Conv2d(inputs, 1, 1, self.df_dim * 8, 1, 1, name='d_template/conv2d', reuse=if_reuse)
        template = conv_batch_normalization(template, 'd_template/batch_norm', is_training=training,
                                            activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        template = Conv2d(template, self.s16, self.s16, 1, self.s16, self.s16, name='d_template/conv2d2',
                          reuse=if_reuse)

        return template


    # Since D is only used during training, we build a template
    # for safe reuse the variables during computing loss for fake/real/wrong images
    # We do not do this for G,
    # because batch_norm needs different options for training and testing
    def get_discriminator(self, x_var, c_var, is_training, no_reuse=None):
        if cfg.GAN.NETWORK_TYPE == "default":
            x_code = self.d_encode_image(training=is_training, inputs=x_var, if_reuse=no_reuse)
            c_code = self.context_embedding(inputs=c_var, if_reuse=no_reuse)
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])
            x_c_code = tf.concat([x_code, c_code], 3)

            return self.discriminator(training=is_training, inputs=x_c_code, if_reuse=no_reuse)

        elif cfg.GAN.NETWORK_TYPE == "simple":
            x_code = self.d_encode_image_simple(training=is_training, inputs=x_var, if_reuse=no_reuse)
            c_code = self.context_embedding(inputs=c_var, if_reuse=no_reuse)
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])
            x_c_code = tf.concat([x_code, c_code], 3)

            return self.discriminator(training=is_training, inputs=x_c_code, if_reuse=no_reuse)
        else:
            raise NotImplementedError
