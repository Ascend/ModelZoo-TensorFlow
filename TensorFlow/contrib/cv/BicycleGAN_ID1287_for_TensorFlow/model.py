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
import os
import time
from tqdm import trange
from layers import *
from folder import check_folder
from imageio import imwrite
import random


class BicycleGAN(object):

    def __init__(self, sess, args):
        self.sess = sess
        self.data_path = args.data_path
        self.checkpoint_dir = os.path.join(args.output_path, 'checkpoints')
        self.result_dir = os.path.join(args.output_path, 'results')
        self.log_dir = os.path.join(args.output_path, 'logs')
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.image_size = args.image_size

        # train
        self.Z_dim = args.Z_dim
        self.learning_rate = args.learning_rate
        self.reconst_coeff = args.reconst_coeff
        self.latent_coeff = args.latent_coeff
        self.kl_coeff = args.kl_coeff

        # test
        self.sample_num = 20  # how many images will model generates for one input

        # Input Image A
        self.image_A = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, 3],
                                      name='input_images')

        # Output Image B
        self.image_B = tf.placeholder(tf.float32, [self.batch_size] + [self.image_size, self.image_size, 3],
                                      name='output_images')

        # Noise z
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.Z_dim], name='latent_vector')

        ''' Implementation of cVAE-GAN: B -> z -> B' '''
        # Encoder is fed the correct output image B for encding it to the latent representation z to learn the distribution of z
        # It outputs 3 things: Enocded value z as Q(z|B), mu of Q(z|B), log_sigma of Q(z|B)
        self.encoded_true_img, self.encoded_mu, self.encoded_log_sigma = self.Encoder(self.image_B)

        # This encoded representation z along with the input image A is then fed to the Generator to output the image B'
        self.desired_gen_img = self.Generator(self.image_A, self.encoded_true_img)  # Image B_cap

        ''' Implementation of cLR-GAN: z -> B' -> z' '''
        # Now, z is sampled from a normal distribution N(z) which in addition to the input image A is fed to the Generator to output B'
        self.LR_desired_img = self.Generator(self.image_A, self.z)  # Generated Image B'

        # B' is then fed to the Encoder to output z' which we try to be close to N(z).
        self.reconst_z, self.reconst_mu, self.reconst_log_sigma = self.Encoder(self.LR_desired_img)  # Encoded z'

        self.P_real = self.Discriminator(self.image_B)  # Probability of ground_truth/real image (B) as real/fake
        self.P_fake = self.Discriminator(
            self.LR_desired_img)  # Probability of generated output images (G(A, N(z)) as real/fake
        self.P_fake_encoded = self.Discriminator(
            self.desired_gen_img)  # Probability of generated output images (G(A, Q(z|B)) as real/fake

        self.loss_vae_gan_D = (tf.reduce_mean(tf.squared_difference(self.P_real, 0.9)) + tf.reduce_mean(
            tf.square(self.P_fake_encoded)))

        self.loss_lr_gan_D = (
                tf.reduce_mean(tf.squared_difference(self.P_real, 0.9)) + tf.reduce_mean(tf.square(self.P_fake)))

        self.loss_vae_gan_GE = tf.reduce_mean(tf.squared_difference(self.P_fake_encoded, 0.9))  # G

        self.loss_gan_G = tf.reduce_mean(tf.squared_difference(self.P_fake, 0.9))

        self.loss_vae_GE = tf.reduce_mean(tf.abs(self.image_B - self.desired_gen_img))  # G

        self.loss_latent_GE = tf.reduce_mean(tf.abs(self.z - self.reconst_z))  # G

        self.loss_kl_E = 0.5 * tf.reduce_mean(
            -1 - self.encoded_log_sigma + self.encoded_mu ** 2 + tf.exp(self.encoded_log_sigma))

        self.loss_D = self.loss_vae_gan_D + self.loss_lr_gan_D - tf.reduce_mean(tf.squared_difference(self.P_real, 0.9))
        self.loss_G = self.loss_vae_gan_GE + self.loss_gan_G + self.reconst_coeff * self.loss_vae_GE + self.latent_coeff * self.loss_latent_GE
        self.loss_E = self.loss_vae_gan_GE + self.reconst_coeff * self.loss_vae_GE + self.latent_coeff * self.loss_latent_GE + self.kl_coeff * self.loss_kl_E

        # Optimizer
        self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        self.gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        self.enc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_solver = opt.minimize(self.loss_D, var_list=self.dis_var)
            self.G_solver = opt.minimize(self.loss_G, var_list=self.gen_var)
            self.E_solver = opt.minimize(self.loss_E, var_list=self.enc_var)

        """ Summary """
        self.d_loss_sum = tf.summary.scalar("d_loss", self.loss_D)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_G)
        self.e_loss_sum = tf.summary.scalar("e_loss", self.loss_E)

    def Discriminator(self, x, is_training=True, reuse=True):
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='d_conv1'))
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv5'), is_training=is_training, scope='d_bn5'))
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv6'), is_training=is_training, scope='d_bn6'))
            x = conv2d_layer(x, 1, 4, 4, 1, 1, name='d_conv7')
            x = tf.reshape(x, [self.batch_size, -1])  # Can use tf.reduce_mean(x, axis=[1, 2, 3])
            x = linear_layer(x, 1, scope='d_fc8')
            x = tf.nn.sigmoid(x)

            return x

    def Generator(self, x, z, is_training=True, reuse=True):
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            conv_layer = []
            z = tf.reshape(z, [self.batch_size, 1, 1, self.Z_dim])
            z = tf.tile(z, [1, self.image_size, self.image_size, 1])
            x = tf.concat([x, z], axis=3)
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='g_conv1'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2'), is_training=is_training, scope='g_bn2'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv3'), is_training=is_training, scope='g_bn3'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv4'), is_training=is_training, scope='g_bn4'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv5'), is_training=is_training, scope='g_bn5'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv6'), is_training=is_training, scope='g_bn6'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv7'), is_training=is_training, scope='g_bn7'))
            conv_layer.append(x)
            x = lrelu_layer(
                bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv8'), is_training=is_training, scope='g_bn8'))

            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv1'), is_training=is_training, scope='gd_bn1'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv2'), is_training=is_training, scope='gd_bn2'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv3'), is_training=is_training, scope='gd_bn3'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv4'), is_training=is_training, scope='gd_bn4'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv5'), is_training=is_training, scope='gd_bn5'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 128, 4, 4, 2, 2, name='g_dconv6'), is_training=is_training, scope='gd_bn6'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 64, 4, 4, 2, 2, name='g_dconv7'), is_training=is_training, scope='gd_bn7'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(
                bn_layer(deconv2d_layer(x, 3, 4, 4, 2, 2, name='g_dconv8'), is_training=is_training, scope='gd_bn8'))
            x = tf.tanh(x)

            return x

    def Encoder(self, x, is_training=True, reuse=True):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='e_conv1'))

            x = residual_block(x, 128, 3, is_training=is_training, name='res_1')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 256, 3, is_training=is_training, name='res_2')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_3')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_4')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = residual_block(x, 512, 3, is_training=is_training, name='res_5')
            x = tf.contrib.layers.avg_pool2d(x, 2, 2, padding='SAME')

            x = tf.contrib.layers.avg_pool2d(x, 8, 8, padding='SAME')
            x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])  # Flattening

            mu = linear_layer(x, self.Z_dim, scope='e_fc1')

            log_sigma = linear_layer(x, self.Z_dim, scope='e_fc2')

            z = mu + tf.random_normal(shape=tf.shape(self.Z_dim)) * tf.exp(log_sigma)

            return z, mu, log_sigma

    def train(self, train_A, train_B):
        # First initialize all variables
        tf.global_variables_initializer().run()

        # saving the model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.num_batches = len(train_A) // self.batch_size

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            counter = 1
            print(" [!] Load failed...")

        # for generating temporary images during training
        self.img_sample = np.expand_dims(train_A[0], axis=0)

        # loop for epoch
        for epoch in range(start_epoch, self.epoch):
            for idx in range(len(train_A)):
                start_time = time.time()

                # get data
                image_A = np.expand_dims(train_A[idx], axis=0)
                image_B = np.expand_dims(train_B[idx], axis=0)
                random_z = np.random.normal(size=(self.batch_size, self.Z_dim))

                _, summary_str_d, D_loss_curr = self.sess.run([self.D_solver, self.d_loss_sum, self.loss_D],
                                                              feed_dict={self.image_A: image_A, self.image_B: image_B,
                                                                         self.z: random_z})
                self.writer.add_summary(summary_str_d, counter)
                _, summary_str_g, G_loss_curr = self.sess.run([self.G_solver, self.g_loss_sum, self.loss_G],
                                                              feed_dict={self.image_A: image_A, self.image_B: image_B,
                                                                         self.z: random_z})
                self.writer.add_summary(summary_str_g, counter)
                _, summary_str_e, E_loss_curr = self.sess.run([self.E_solver, self.e_loss_sum, self.loss_E],
                                                              feed_dict={self.image_A: image_A, self.image_B: image_B,
                                                                         self.z: random_z})
                self.writer.add_summary(summary_str_e, counter)

                # Saving training results for every 100 examples
                temp_dir = check_folder(os.path.join(self.result_dir, 'temps'))
                if counter % 100 == 0:
                    z_sample = np.random.normal(size=(1, self.Z_dim))
                    samples = self.sess.run(self.LR_desired_img,
                                            feed_dict={self.image_A: self.img_sample, self.z: z_sample})
                    # transform from [-1,1] to [0,255]
                    samples = (np.asarray(samples + 1) / 2 * 255).astype(np.uint8)
                    imwrite(os.path.join(temp_dir, f'train_{epoch}_{idx}.jpg'),
                            np.squeeze(samples))

                # display training status
                counter += 1
                cost_time = time.time() - start_time
                print("epoch : {}----step : {}----|d_loss : {}----g_loss : {}----e_loss : {}|----sec/step : {}"
                      .format(epoch, counter, D_loss_curr, G_loss_curr, E_loss_curr, cost_time))

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def test(self, test_A, test_B):  # generate images
        self.step = 0

        for idx in trange(len(test_A)):
            self.step += 1
            save_dir = check_folder(os.path.join(self.result_dir, "test_results", str(self.step)))

            # get input and save groundtruth
            image_A = np.expand_dims(test_A[idx], axis=0)
            imwrite(os.path.join(save_dir, f'ground_truth.jpg'),
                    (np.asarray(test_B[idx] + 1) / 2 * 255).astype(np.uint8))

            # generate images
            for i in range(0, self.sample_num):
                z = np.random.normal(size=(1, self.Z_dim))
                LR_desired_img = self.sess.run(self.LR_desired_img,
                                               feed_dict={self.image_A: image_A, self.z: z})
                # transform from [-1,1] to [0,255]
                LR_desired_img = (np.asarray(LR_desired_img + 1) / 2 * 255).astype(np.uint8)
                imwrite(os.path.join(save_dir, f'random_{i + 1}.jpg'),
                        np.squeeze(LR_desired_img))

    def save(self, checkpoint_dir, step):  # save checkpoints
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'BicycleGAN.model'), global_step=step)

    def load(self, checkpoint_dir):  # load checkpoint if it exits
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Successful in reading {}".format(ckpt_name))
            return True, counter
        else:
            print(" [!] Failed to find checkpoint directory")
            return False, 0
