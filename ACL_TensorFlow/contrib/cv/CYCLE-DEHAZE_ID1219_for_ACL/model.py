# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
"""
  This file is used to describe the CYCLEGAN model.
"""
from npu_bridge.npu_init import *
import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator

import vgg16

REAL_LABEL = 0.9


class CycleGAN:
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
  """
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 batch_size=1,
                 image_size1=256,
                 image_size2=256,
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10.0,
                 lambda2=10.0,
                 learning_rate=1e-4,
                 beta1=0.5,
                 ngf=64):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file

        self.is_training = tf.placeholder_with_default(True,
                                                       shape=[],
                                                       name='is_training')

        self.G = Generator('G',
                           self.is_training,
                           ngf=ngf,
                           norm=norm,
                           image_size1=image_size1,
                           image_size2=image_size2)
        self.D_Y = Discriminator('D_Y',
                                 self.is_training,
                                 norm=norm,
                                 use_sigmoid=use_sigmoid)
        self.F = Generator('F',
                           self.is_training,
                           ngf=ngf,
                           norm=norm,
                           image_size1=image_size1,
                           image_size2=image_size2)
        self.D_X = Discriminator('D_X',
                                 self.is_training,
                                 norm=norm,
                                 use_sigmoid=use_sigmoid)

        self.fake_x = tf.placeholder(
            tf.float32, shape=[batch_size, image_size1, image_size2, 3])
        self.fake_y = tf.placeholder(
            tf.float32, shape=[batch_size, image_size1, image_size2, 3])

        self.vgg = vgg16.Vgg16()

    def model(self):
        '''
    create model
    return loss
    '''
        X_reader = Reader(self.X_train_file,
                          name='X',
                          image_size1=self.image_size1,
                          image_size2=self.image_size2,
                          batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file,
                          name='Y',
                          image_size1=self.image_size1,
                          image_size2=self.image_size2,
                          batch_size=self.batch_size)

        x = X_reader.feed()
        y = Y_reader.feed()

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
        perceptual_loss = self.perceptual_similarity_loss(
            self.G, self.F, x, y, self.vgg)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y,
                                         fake_y,
                                         use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss + perceptual_loss  # + pixel_loss
        D_Y_loss = self.discriminator_loss(self.D_Y,
                                           y,
                                           self.fake_y,
                                           use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = self.generator_loss(self.D_X,
                                         fake_x,
                                         use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss + perceptual_loss  # + pixel_loss
        D_X_loss = self.discriminator_loss(self.D_X,
                                           x,
                                           self.fake_x,
                                           use_lsgan=self.use_lsgan)

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/perceptual_loss', perceptual_loss)
        # tf.summary.scalar('loss/pixel_loss', pixel_loss)

        tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
        tf.summary.image('X/reconstruction',
                         utils.batch_convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction',
                         utils.batch_convert2int(self.G(self.F(y))))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        '''
    Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
    and a linearly decaying rate that goes to zero over the next 100k steps
    '''
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (tf.where(
                tf.greater_equal(global_step, start_decay_step),
                tf.train.polynomial_decay(starter_learning_rate,
                                          global_step - start_decay_step,
                                          decay_steps,
                                          end_learning_rate,
                                          power=1.0), starter_learning_rate))
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (npu_distributed_optimizer_wrapper(
                tf.train.AdamOptimizer(learning_rate, beta1=beta1,
                                       name=name)).minimize(
                                           loss,
                                           global_step=global_step,
                                           var_list=variables))
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss,
                                       self.D_Y.variables,
                                       name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss,
                                       self.D_X.variables,
                                       name='Adam_D_X')
        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
        if use_lsgan:
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(
                D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y, use_lsgan=True):
        """  fool discriminator into believing that G(x) is real
    """
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
    """
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def perceptual_similarity_loss(self, G, F, x, y, vgg):
        '''
    perceptual similarity loss
    '''
        x1 = tf.image.resize_images(x, [224, 224])  # to feed vgg, need resize
        y1 = tf.image.resize_images(y, [224, 224])

        rx = F(G(x))  # create reconstructed images
        ry = G(F(y))

        rx1 = tf.image.resize_images(rx,
                                     [224, 224])  # to feed vgg, need resize
        ry1 = tf.image.resize_images(ry, [224, 224])

        fx1, fx2 = vgg.build(x1)  # extract features from vgg
        fy1, fy2 = vgg.build(y1)

        frx1, frx2 = vgg.build(
            rx1)  # extract features from vgg (2nd pool & 5th pool
        fry1, fry2 = vgg.build(ry1)

        m1 = tf.reduce_mean(tf.squared_difference(fx1, frx1))  # mse difference
        m2 = tf.reduce_mean(tf.squared_difference(fx2, frx2))

        m3 = tf.reduce_mean(tf.squared_difference(fy1, fry1))
        m4 = tf.reduce_mean(tf.squared_difference(fy2, fry2))

        perceptual_loss = (
            m1 + m2 + m3 + m4
        ) * 0.00001 * 0.5  # calculate perceptual loss and give weight (0.00001*0.5)
        return perceptual_loss

# def pixel_wise_loss(self, G, F, x, y):
#   rx = F(G(x))
#   ry = G(F(y))
#   pixel_wise_loss = tf.reduce_mean(tf.squared_difference(x, rx)) + tf.reduce_mean(tf.squared_difference(y, ry))
#   return 10*pixel_wise_loss
