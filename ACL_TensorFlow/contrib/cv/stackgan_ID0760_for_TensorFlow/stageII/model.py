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

from custom_ops import fc, conv_batch_normalization, fc_batch_normalization, reshape, Conv2d, UpSample, add
from config import cfg


class CondGAN(object):
    def __init__(self, lr_imsize, hr_lr_ratio):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.hr_lr_ratio = hr_lr_ratio
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.s = lr_imsize
        print('lr_imsize: ', lr_imsize)
        self.s2, self.s4, self.s8, self.s16 = int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

    # conditioning augmentation structure for text embedding
    # are shared by g and hr_g
    # g and hr_g build this structure separately and do not share parameters
    def generate_condition(self, c_var):
        conditions = fc(c_var, self.ef_dim * 2, 'gen_cond/fc', activation_fn=tf.nn.leaky_relu)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    # stage I generator (g)
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
        node2_1 = conv_batch_normalization(node2_1, 'g_n2.1/batch_norm', activation_fn=tf.nn.relu,
                                           is_training=training)
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

    def get_generator(self, z_var, is_training):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var, training=is_training)
        else:
            raise NotImplementedError

    # stage II generator (hr_g)
    def residual_block(self, x_c_code, name, training=True):
        node0_0 = x_c_code  # -->s4 * s4 * gf_dim * 4

        node0_1 = Conv2d(x_c_code, 3, 3, self.gf_dim * 4, 1, 1, name=name+'/conv2d')
        node0_1 = conv_batch_normalization(node0_1, name+'/batch_norm', is_training=training,
                                           activation_fn=tf.nn.relu)
        node0_1 = Conv2d(node0_1, 3, 3, self.gf_dim * 4, 1, 1, name=name+'/conv2d2')
        node0_1 = conv_batch_normalization(node0_1, name+'/batch_norm2', is_training=training)

        output_tensor = add([node0_0, node0_1], name='resid_block/add')
        output_tensor = tf.nn.relu(output_tensor)

        return output_tensor

    def hr_g_encode_image(self, x_var, training=True):  # input: x_var --> s * s * 3
        # s * s * gf_dim
        output_tensor = Conv2d(x_var, 3, 3, self.gf_dim, 1, 1, activation_fn=tf.nn.relu, name='hr_g_OT/conv2d')

        # s2 * s2 * gf_dim * 2
        output_tensor = Conv2d(output_tensor, 4, 4, self.gf_dim * 2, 2, 2, name='hr_g_OT/conv2d2')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_g_OT/batch_norm', is_training=training,
                                                 activation_fn=tf.nn.relu)
        # s4 * s4 * gf_dim * 4
        output_tensor = Conv2d(output_tensor, 4, 4, self.gf_dim * 4, 2, 2, name='hr_g_OT/conv2d3')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_g_OT/batch_norm2', is_training=training,
                                                 activation_fn=tf.nn.relu)
        return output_tensor

    def hr_g_joint_img_text(self, x_c_code, training=True):  # input: x_code: -->s4 * s4 * (ef_dim+gf_dim*4)
        # s4 * s4 * gf_dim * 4
        output_tensor = Conv2d(x_c_code, 3, 3, self.gf_dim * 4, 1, 1, name='hr_g_joint_OT/conv2d')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_g_joint_OT/batch_norm', is_training=training,
                                                 activation_fn=tf.nn.relu)
        return output_tensor

    def hr_generator(self, x_c_code, training=True):  # Input: x_c_code -->s4 * s4 * gf_dim*4
        output_tensor = UpSample(x_c_code, size=[self.s2, self.s2], method=1, align_corners=False,
                                 name='hr_gen/upsample')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim * 2, 1, 1, name='hr_gen/conv2d')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_gen/batch_norm', is_training=training,
                                                 activation_fn=tf.nn.relu)
        output_tensor = UpSample(output_tensor, size=[self.s, self.s], method=1, align_corners=False,
                                 name='hr_gen/upsample2')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim, 1, 1, name='hr_gen/conv2d2')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_gen/batch_norm2', is_training=training,
                                                 activation_fn=tf.nn.relu)
        output_tensor = UpSample(output_tensor, size=[self.s * 2, self.s * 2], method=1, align_corners=False,
                                 name='hr_gen/upsample3')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim//2, 1, 1, name='hr_gen/conv2d3')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_gen/batch_norm3', is_training=training,
                                                 activation_fn=tf.nn.relu)
        output_tensor = UpSample(output_tensor, size=[self.s * 4, self.s * 4], method=1, align_corners=False,
                                 name='hr_gen/upsample3')
        output_tensor = Conv2d(output_tensor, 3, 3, self.gf_dim//4, 1, 1, name='hr_gen/conv2d4')
        output_tensor = conv_batch_normalization(output_tensor, 'hr_gen/batch_norm4', is_training=training,
                                                 activation_fn=tf.nn.relu)
        # -->4s * 4s * 3
        output_tensor = Conv2d(output_tensor, 3, 3, 3, 1, 1, name='hr_gen/conv2d5', activation_fn=tf.nn.tanh)
        return output_tensor

    def hr_get_generator(self, x_var, c_code, is_training):
        if cfg.GAN.NETWORK_TYPE == "default":
            # image x_var: self.s * self.s *3
            x_code = self.hr_g_encode_image(x_var, training=is_training)  # -->s4 * s4 * gf_dim * 4

            # text c_code: ef_dim
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s4, self.s4, 1])

            # combine both --> s4 * s4 * (ef_dim+gf_dim*4)
            x_c_code = tf.concat([x_code, c_code], 3)

            # Joint learning from text and image -->s4 * s4 * gf_dim * 4
            node0 = self.hr_g_joint_img_text(x_c_code)
            node1 = self.residual_block(node0, 'node1_resid_block', training=is_training)
            node2 = self.residual_block(node1, 'node2_resid_block', training=is_training)
            node3 = self.residual_block(node2, 'node3_resid_block', training=is_training)
            node4 = self.residual_block(node3, 'node4_resid_block', training=is_training)

            # Up-sampling
            return self.hr_generator(node4, training=is_training)  # -->4s * 4s * 3
        else:
            raise NotImplementedError

    # structure shared by d and hr_d
    # d and hr_d build this structure separately and do not share parameters
    def context_embedding(self, inputs=None, if_reuse=None):
        template = fc(inputs, self.ef_dim, 'd_embedd/fc', activation_fn=tf.nn.leaky_relu, reuse=if_reuse)

        return template

    def discriminator(self, training=True, inputs=None, if_reuse=None):
        template = Conv2d(inputs, 1, 1, self.df_dim * 8, 1, 1, name='d_template/conv2d', reuse=if_reuse)
        template = conv_batch_normalization(template, 'd_template/batch_norm', is_training=training,
                                            activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        template = Conv2d(template, self.s16, self.s16, 1, self.s16, self.s16, name='d_template/conv2d2',
                          reuse=if_reuse)

        return template

    # d-net
    def d_encode_image(self, inputs=None, training=True, if_reuse=None):
        # input: s * s * 3
        node1_0 = Conv2d(inputs, 4, 4, self.df_dim, 2, 2, activation_fn=tf.nn.leaky_relu, name='d_n1.0/conv2d',
                         reuse=if_reuse)  # s2 * s2 * df_dim

        # s4 * s4 * df_dim*2
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 2, 2, 2, name='d_n1.0/conv2d2', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'd_n1.0/batch_norm', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s8 * s8 * df_dim*4
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 4, 2, 2, name='d_n1.0/conv2d3', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'd_n1.0/batch_norm2', is_training=training, reuse=if_reuse)
        # s16 * s16 * df_dim*8
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

        node1 = add([node1_0, node1_1], name='d_n1/add')
        node1 = tf.nn.leaky_relu(node1)

        return node1

    def get_discriminator(self, x_var, c_var, is_training, no_reuse=None):
        if cfg.GAN.NETWORK_TYPE == "default":
            x_code = self.d_encode_image(training=is_training, inputs=x_var, if_reuse=no_reuse)  # s16 * s16 * df_dim*8

            c_code = self.context_embedding(inputs=c_var, if_reuse=no_reuse)
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim

            x_c_code = tf.concat([x_code, c_code], 3)
            return self.discriminator(training=is_training, inputs=x_c_code, if_reuse=no_reuse)
        else:
            raise NotImplementedError

    # hr_d_net
    def hr_d_encode_image(self, inputs=None, training=True, if_reuse=None):
        #  input:  4s * 4s * 3
        node1_0 = Conv2d(inputs, 4, 4, self.df_dim, 2, 2, activation_fn=tf.nn.leaky_relu,
                         name='hr_d_encode_n1.0/conv2d1', reuse=if_reuse)  # 2s * 2s * df_dim

        # s * s * df_dim*2
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 2, 2, 2, name='hr_d_encode_n1.0/conv2d2', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s2 * s2 * df_dim*4
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 4, 2, 2, name='hr_d_encode_n1.0/conv2d3', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm2', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s4 * s4 * df_dim*8
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 8, 2, 2, name='hr_d_encode_n1.0/conv2d4', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm3', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s8 * s8 * df_dim*16
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 16, 2, 2, name='hr_d_encode_n1.0/conv2d5', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm4', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s16 * s16 * df_dim*32
        node1_0 = Conv2d(node1_0, 4, 4, self.df_dim * 32, 2, 2, name='hr_d_encode_n1.0/conv2d6', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm5', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s16 * s16 * df_dim*16
        node1_0 = Conv2d(node1_0, 1, 1, self.df_dim * 16, 1, 1, name='hr_d_encode_n1.0/conv2d7', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm6', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        # s16 * s16 * df_dim*8
        node1_0 = Conv2d(node1_0, 1, 1, self.df_dim * 8, 1, 1, name='hr_d_encode_n1.0/conv2d8', reuse=if_reuse)
        node1_0 = conv_batch_normalization(node1_0, 'hr_d_encode_n1.0/batch_norm7', is_training=training,
                                           reuse=if_reuse)

        node1_1 = Conv2d(node1_0, 1, 1, self.df_dim * 2, 1, 1, name='hr_d_encode_n1.1/conv2d', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'hr_d_encode_n1.1/batch_norm', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        node1_1 = Conv2d(node1_1, 3, 3, self.df_dim * 2, 1, 1, name='hr_d_encode_n1.1/conv2d2', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'hr_d_encode_n1.1/batch_norm2', is_training=training,
                                           activation_fn=tf.nn.leaky_relu, reuse=if_reuse)
        node1_1 = Conv2d(node1_1, 3, 3, self.df_dim * 8, 1, 1, name='hr_d_encode_n1.1/conv2d3', reuse=if_reuse)
        node1_1 = conv_batch_normalization(node1_1, 'hr_d_encode_n1.1/batch_norm3', is_training=training,
                                           reuse=if_reuse)

        node1 = add([node1_0, node1_1], name='hr_d_encode_n1/add')
        node1 = tf.nn.leaky_relu(node1)

        return node1

    def hr_get_discriminator(self, x_var, c_var, is_training, no_reuse=None):
        if cfg.GAN.NETWORK_TYPE == "default":
            # s16 * s16 * df_dim*8
            x_code = self.hr_d_encode_image(training=is_training, inputs=x_var, if_reuse=no_reuse)

            c_code = self.context_embedding(inputs=c_var, if_reuse=no_reuse)
            c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
            c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])  # s16 * s16 * ef_dim

            x_c_code = tf.concat([x_code, c_code], 3)
            return self.discriminator(training=is_training, inputs=x_c_code, if_reuse=no_reuse)
        else:
            raise NotImplementedError
