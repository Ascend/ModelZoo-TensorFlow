
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

# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
# import scipy.misc
from PIL import Image
from layers import *
# from tensorflow.contrib import layers
from folder_npu import check_folder
from load_data_npu import load_images, save_images, imsave, load_batch_image, load_test_image
import matplotlib

matplotlib.use('Tkagg')
#import matplotlib.pyplot as plt
#import moxing as mox

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BicycleGAN(object):
    model_name = "BicycleGAN"

    def __init__(self, sess, epoch, batch_size, Z_dim, image_size, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.TMP_DATA_PATH = dataset_name
        self.TMP_RESULTS_PATH = result_dir
        self.TMP_CHECKPOINT_PATH = checkpoint_dir
        self.TMP_LOGS_PATH = log_dir
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.image_size = image_size

        self.input_width = 256
        self.input_height = 256
        self.output_width = 256
        self.output_height = 256
        self.channels = 3

        self.Z_dim = Z_dim

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.reconst_coeff = 10
        self.latent_coeff = 0.5
        self.kl_coeff = 0.01

        # test
        self.sample_num = 64

        # load data
        # self.train_A, self.train_B, self.test_A, self.test_B = load_images()
        #self.train_A = mox.file.glob(dataset_name + "/train/*.jpg")
        self.train_A = glob.glob(dataset_name + "/train/*.jpg")
        self.num_batches = len(self.train_A) // self.batch_size

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

    def build_model(self):
        image_dims = [self.input_width, self.input_height, self.channels]

        ''' Graph input '''
        # Input Image A
        self.image_A = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='input_images')

        # Output Image B
        self.image_B = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='output_images')

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

        """ Testing """
        # self.fake_images = self.Generator(self.image_A, self.z, is_training=False, reuse=True)

        """ Summary """

        self.d_loss_sum = tf.summary.scalar("d_loss", self.loss_D)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.loss_G)
        self.e_loss_sum = tf.summary.scalar("e_loss", self.loss_E)

    # final summary operations
    # self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    # self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    # self.q_sum = tf.summary.merge([q_loss_sum, q_disc_sum, q_cont_sum])

    def train(self):

        # include code for logger.info()

        # First initialize all variables
        tf.global_variables_initializer().run()

        # Input to graph from training data
        self.z_sample = np.random.normal(size=(self.batch_size, self.Z_dim))
        input_img1, batch_imagesB = load_batch_image(0, self.dataset_name)
        self.input_img1 = np.expand_dims(input_img1, axis=0)
        # self.input_img1 = self.train_A[0:self.batch_size] # training results for a single image
        # saving the model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(len(self.train_A)):
                start_time = time.time()
                batch_imagesA, batch_imagesB = load_batch_image(idx, self.dataset_name)
                batch_imagesA = np.expand_dims(batch_imagesA, axis=0)
                batch_imagesB = np.expand_dims(batch_imagesB, axis=0)
                batch_z = np.random.normal(size=(self.batch_size, self.Z_dim))

                _, summary_str_d, D_loss_curr = self.sess.run([self.D_solver, self.d_loss_sum, self.loss_D],
                                                              feed_dict={self.image_A: batch_imagesA,
                                                                         self.image_B: batch_imagesB, self.z: batch_z})
                self.writer.add_summary(summary_str_d, counter)
                _, summary_str_g, G_loss_curr = self.sess.run([self.G_solver, self.g_loss_sum, self.loss_G],
                                                              feed_dict={self.image_A: batch_imagesA,
                                                                         self.image_B: batch_imagesB, self.z: batch_z})
                self.writer.add_summary(summary_str_g, counter)
                _, summary_str_e, E_loss_curr = self.sess.run([self.E_solver, self.e_loss_sum, self.loss_E],
                                                              feed_dict={self.image_A: batch_imagesA,
                                                                         self.image_B: batch_imagesB, self.z: batch_z})
                self.writer.add_summary(summary_str_e, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f  d_loss: %.8f g_loss: %.8f e_loss: %.8f" % (
                    epoch, idx, self.num_batches, time.time() - start_time, D_loss_curr, G_loss_curr, E_loss_curr))

                # Saving training results for every 100 examples
                if counter % 100 == 0:
                    samples = self.sess.run(self.LR_desired_img,
                                            feed_dict={self.image_A: self.input_img1, self.z: self.z_sample})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                        epoch, idx))
            # mox.file.copy_parallel(self.TMP_RESULTS_PATH, self.OBS_RESULTS_PATH)
            # mox.file.copy_parallel(self.TMP_DATA_PATH, self.OBS_DATA_PATH)
            # mox.file.copy_parallel(self.TMP_LOGS_PATH, self.OBS_LOG_PATH)
            # mox.file.copy_parallel(self.TMP_CHECKPOINT_PATH, self.OBS_CHECKPOINT_DIR)
            # After an epoch, start_batch_id is set to zero
            start_batch_id = 0
            # non-zero value is only for the first epoch after loading pre-trained model
            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def test(self):
        self.step = 0
        base_dir = os.path.join('test_results')
        check_folder(os.path.join(self.result_dir, base_dir))
        #test_all = mox.file.glob(self.dataset_name + "/val/*.jpg")
        test_all = glob.glob(self.dataset_name + "/val/*.jpg")
        for idx in range(len(test_all)):
            self.step += 1
            img_A = load_test_image(idx, self.dataset_name)
            input_img = np.expand_dims(img_A, axis=0)
            z = np.random.normal(size=(self.batch_size, self.Z_dim))
            LR_desired_img = self.sess.run(self.LR_desired_img, feed_dict={self.image_A: input_img, self.z: z})
            image = LR_desired_img[0]
            image = Image.fromarray(np.uint8((np.concatenate((img_A * 255., image * 255.), axis=1))))
            image.save(os.path.join(self.result_dir, base_dir, 'random_{}.jpg'.format(self.step)))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.model_name, (self.dataset_name).split("/")[-1], self.batch_size, self.Z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            #mox.file.make_dirs(checkpoint_dir)
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

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
