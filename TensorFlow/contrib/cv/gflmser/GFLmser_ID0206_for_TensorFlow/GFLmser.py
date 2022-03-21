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
from tensorflow.python.compat import compat


class GFLmser():
    def __init__(self, config):
        self.config = config
        self.input_data = tf.placeholder(tf.float32, [config.batch_size, 64, 64, 3])
        self.lr_img = tf.placeholder(tf.float32, [config.batch_size, 16, 16, 3])
        self.hr_img = tf.placeholder(tf.float32, [config.batch_size, 64, 64, 3])
        self.real_lr = tf.placeholder(tf.float32, [config.batch_size, 16, 16, 3])
        self.training = tf.placeholder(tf.bool)
        tf.summary.image('input_img', tf.clip_by_value(self.input_data[:1]*255, 0, 255))
        tf.summary.image('lr_img', tf.clip_by_value(self.lr_img[:1]*255, 0, 255))
        tf.summary.image('label', tf.clip_by_value(self.hr_img[:1]*255, 0, 255))


        self.global_steps_up = tf.Variable(0, trainable=False)
        self.global_steps_down = tf.Variable(0, trainable=False)
        self.global_steps_dis = tf.Variable(0, trainable=False)

        self.build_model()
        self.build_optimizer()

    def build_model(self):
        self.fake_lr, out_list = self.high_low(self.input_data)
        self.real_out = self.discriminator(self.real_lr)
        self.fake_out = self.discriminator(self.fake_lr)
        self.train_out_hr = self.low_high(self.fake_lr, out_list)
        self.test_out_hr = self.low_high(self.lr_img, out_list)
        tf.summary.image('fake_lr', tf.clip_by_value(self.fake_lr[:1]*255, 0, 255))
        tf.summary.image('fake_hr', tf.clip_by_value(self.train_out_hr[:1]*255, 0, 255))
        tf.summary.image('test_hr', tf.clip_by_value(self.test_out_hr[:1]*255, 0, 255))
        self.PSNR = tf.reduce_mean(tf.image.psnr(self.test_out_hr, self.hr_img, max_val=1.0))
        tf.summary.scalar('psnr', self.PSNR)

    def build_optimizer(self):
        self.discrim_cost_real = tf.reduce_mean(tf.square(self.real_out - tf.ones_like(self.real_out)))
        self.discrim_cost_fake = tf.reduce_mean(tf.square(self.fake_out - tf.zeros_like(self.fake_out)))
        self.discrim_cost = tf.reduce_mean(self.discrim_cost_real + self.discrim_cost_fake)

        self.up_mse_loss = tf.reduce_mean(tf.square(self.fake_lr - self.lr_img))
        self.generator_loss = tf.reduce_mean(tf.square(self.fake_out - tf.ones_like(self.fake_out)))
        self.generator_cost = self.up_mse_loss + self.generator_loss * self.config.beta

        self.down_mse_loss = tf.reduce_mean(tf.square(self.train_out_hr - self.hr_img))

        tf.summary.scalar("d_cost", self.discrim_cost)
        tf.summary.scalar("up_mse_cost", self.up_mse_loss)
        tf.summary.scalar("generator_cost", self.generator_cost)
        tf.summary.scalar("down_mse_loss", self.down_mse_loss)
        dis_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)
        up_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)
        down_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)

        up_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='high2low')
        dis_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        down_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='low2high')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.dis_train_op = dis_optimizer.minimize(self.discrim_cost, var_list=dis_params, global_step=self.global_steps_dis)
            self.up_train_op = up_optimizer.minimize(self.generator_cost, var_list=up_params, global_step=self.global_steps_up)
            self.down_train_op = down_optimizer.minimize(self.down_mse_loss, var_list=down_params, global_step=self.global_steps_down)
            # self.all_train_op = tf.group(self.gen_train_op, self.dis_train_op)

    def high_low(self, input_img):
        with tf.variable_scope('high2low', reuse=tf.AUTO_REUSE):
            out_list = []
            out_list.append(input_img)
            blocks = [96, 96, 128, 128, 256, 256, 512, 512, 128, 128, 32, 32]
            out = tf.layers.conv2d(input_img, filters=blocks[0], kernel_size=3, strides=1, padding='same',
                                   use_bias=False)  # [-1,64,64,96]

            # for i in range(len(self.downs)):
            #     out = self.downs[i](out)
            #     if i % 2 == 1:
            #         out_list.append(out)

            out = self.BasicBlock(out, blocks[0], blocks[0], downsample=not 0 % 2)
            for i in range(1, 8):
                out = self.BasicBlock(out, blocks[i - 1], blocks[i], downsample=not i % 2)
                if i % 2 == 1:
                    out_list.append(out)

            # for j in range(len(self.ups)):
            #     out = self.ups[j](out)
            #     if j % 3 == 2:
            #         out_list.append(out)

            for j in range(2):
                out = tf.nn.depth_to_space(out, 2)
                out = self.BasicBlock(out, blocks[8 + j * 2], blocks[8 + j * 2])
                out = self.BasicBlock(out, blocks[9 + j * 2], blocks[9 + j * 2])
                out_list.append(out)

            out = tf.layers.conv2d(out, 8, kernel_size=3, strides=1, padding='same', use_bias=False)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, 3, kernel_size=3, strides=1, padding='same', use_bias=False)
            out = tf.nn.tanh(out)

            out = tf.clip_by_value((out + 1) / 2, 0, 1)
            # print('high2low.shape=', out.shape)
            return out, out_list

    def low_high(self, input_img, up_list):
        with tf.variable_scope('low2high', reuse=tf.AUTO_REUSE):
            blocks = [128, 128, 512, 512, 256, 256, 128, 128, 96, 96, 32, 32]

            # in_layer
            out = tf.layers.conv2d(input_img, 96, kernel_size=3, strides=1, padding='same', use_bias=False)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, blocks[0], kernel_size=3, strides=1, padding='same', use_bias=False)
            out = tf.nn.relu(out)

            # down
            out = self.BasicBlock(out, inplanes=blocks[0], planes=blocks[0], downsample=not 0 % 2)
            out = self.BasicBlock(out, inplanes=blocks[0], planes=blocks[1], downsample=not 1 % 2)
            out = out + up_list[5]
            out = self.BasicBlock(out, inplanes=blocks[1], planes=blocks[2], downsample=not 2 % 2)
            out = self.BasicBlock(out, inplanes=blocks[2], planes=blocks[3], downsample=not 3 % 2)
            out = out + up_list[4]

            # up
            list_index = 3
            for j in range(4):
                out = tf.layers.conv2d_transpose(out, filters=blocks[4 + j * 2], kernel_size=4, strides=2,
                                                 padding='same')
                out = self.BasicBlock(out, blocks[4 + j * 2], blocks[4 + j * 2])
                out = self.BasicBlock(out, blocks[5 + j * 2], blocks[5 + j * 2])
                if list_index > 0:
                    out = out + up_list[list_index]
                    list_index -= 1

            # out_layer
            in_feat = blocks[-1]
            out = tf.layers.conv2d(out, in_feat, kernel_size=3, strides=1, padding='same', use_bias=False)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, 3, kernel_size=1, strides=1, padding='VALID')
            out = tf.nn.tanh(out)

            # out_layer_merge
            out = tf.layers.conv2d((out + 1) / 2 + up_list[0], 3, kernel_size=1, strides=1, padding='VALID')
            out = tf.nn.tanh(out)

            out = tf.clip_by_value((out + 1) / 2, 0, 1)
            return out

    def discriminator(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            blocks = [128, 128, 256, 256, 512, 512]
            pool_start = len(blocks) - 2
            out = x
            in_feat = 3
            for i in range(len(blocks)):
                b_down = bool(i >= pool_start)
                out = self.BasicBlock_discrim(out, in_feat, blocks[i], downsample=b_down, nobn=True)
                in_feat = blocks[i]
            out = tf.reshape(out, (-1, 16 * blocks[-1]))
            # out_layer
            out = tf.layers.dense(out, units=blocks[-1])
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(out, training=self.training)
            out = tf.nn.relu(out)
            out = tf.layers.dense(out, units=1, activation=tf.nn.sigmoid)
            return out

    def BasicBlock(self, x, inplanes, planes, stride=1, downsample=False, upsample=False, nobn=False):
        residual = x
        if not nobn:
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(x, training=self.training)
            out = tf.nn.relu(out)
        else:
            out = tf.nn.relu(x)
        if upsample:
            out = tf.layers.conv2d_transpose(out, filters=planes, kernel_size=4, strides=2, padding='same')
        else:
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        if not nobn:
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(out, training=self.training)
        out = tf.nn.relu(out)
        if downsample:
            out = tf.nn.avg_pool2d(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        else:
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        if inplanes != planes or upsample or downsample:
            if upsample:
                skip = tf.layers.conv2d_transpose(x, filters=planes, kernel_size=4, strides=2, padding='same')
            elif downsample:
                skip = tf.nn.avg_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                skip = tf.layers.conv2d(skip, filters=planes, kernel_size=1, strides=1, padding='VALID')
            else:
                skip = tf.layers.conv2d(x, filters=planes, kernel_size=1, strides=1, padding='VALID')
        else:
            skip = None
        if skip is not None:
            residual = skip
        out += residual
        return out

    def BasicBlock_discrim(self, x, inplanes, planes, stride=1, downsample=False, nobn=False):
        residual = x
        if not nobn:
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(x, training=self.training)
            out = tf.nn.relu(out)
        else:
            out = tf.nn.relu(x)

        out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        with compat.forward_compatibility_horizon(2019, 5, 1):
            out = tf.layers.batch_normalization(out, training=self.training)

        if not nobn:
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(out, training=self.training)
        out = tf.nn.relu(out)

        if downsample:
            out = tf.nn.avg_pool2d(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(out, training=self.training)
        else:
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
            with compat.forward_compatibility_horizon(2019, 5, 1):
                out = tf.layers.batch_normalization(out, training=self.training)

        if inplanes != planes or downsample:
            if downsample:
                skip = tf.nn.avg_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                skip = tf.layers.conv2d(skip, filters=planes, kernel_size=1, strides=1, padding='VALID')
                with compat.forward_compatibility_horizon(2019, 5, 1):
                    skip = tf.layers.batch_normalization(skip, training=self.training)
            else:
                skip = tf.layers.conv2d(x, filters=planes, kernel_size=1, strides=1, padding='VALID')
                with compat.forward_compatibility_horizon(2019, 5, 1):
                    skip = tf.layers.batch_normalization(skip, training=self.training)
        else:
            skip = None
        if skip is not None:
            residual = skip
        out += residual
        return out
