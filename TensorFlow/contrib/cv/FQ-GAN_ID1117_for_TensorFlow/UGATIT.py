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
# ============================================================================

from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from vq_layer import VectorQuantizerEMA
import shutil
#transfor
#from npu_bridge.npu_init import *

class UGATIT(object) :
    def __init__(self, sess, args):
        self.light = args.light
        self.if_quant = args.quant
        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.ld = args.GP_ld
        self.smoothing = args.smoothing

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.test_train = args.test_train

        if self.if_quant:
            self.commitment_cost = args.commitment_cost
        else:
            self.commitment_cost = 0.0
        layerwise_channel = [64, 128, 256, 512, 1024, 2028]
        
        # 
        num_embed = [5, 6, 7, 7, 7, 7]
#         num_embed = [5, 6, 7, 8, 9, 10]
        self.quantization_layer = args.quantization_layer
        self.quant_layers = [int(x) for x in args.quantization_layer]

        self.decay = args.decay


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)
        # self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name, size=self.img_size
        #self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        #self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainA_dataset = glob('{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('{}/*.*'.format(self.dataset_name + '/trainB'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        self.quantize = {}
        for layer in self.quant_layers:
            self.quantize[layer] = VectorQuantizerEMA(embedding_dim=layerwise_channel[layer],
                                               num_embeddings=2**num_embed[layer],
                                               commitment_cost=self.commitment_cost, decay=self.decay)
        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# smoothing : ", self.smoothing)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)


    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing :
            smoothing = '_smoothing'
        else :
            smoothing = ''

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        if not self.if_quant:
            return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}".format(self.model_name, self.dataset_name,
                                                             self.gan_type, n_res, n_dis,
                                                             self.n_critic,
                                                             self.adv_weight, self.cycle_weight,
                                                              self.identity_weight, self.cam_weight,
                                                              sn, smoothing)
        else:
            return "{}_q_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}_{}_{}_{}".format(self.model_name,
                                                                  self.dataset_name,
                                                             self.gan_type, n_res, n_dis,
                                                             self.n_critic,
                                                             self.adv_weight, self.cycle_weight,
                                                              self.identity_weight, self.cam_weight,
                                                              sn, smoothing, self.quantization_layer,
                                                              self.commitment_cost, self.decay)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            # Down-Sampling Bottleneck
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # Class Activation Map
            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = relu(x)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            # Gamma, Beta block
            gamma, beta = self.MLP(x, reuse=reuse)

            # Up-Sampling Bottleneck
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(x, channel, gamma, beta, smoothing=self.smoothing, scope='adaptive_resblock' + str(i))

            # Up-Sampling
            for i in range(2) :
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='up_conv_'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = relu(x)

                channel = channel // 2


            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x, cam_logit, heatmap

    def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
        channel = self.ch * self.n_res

        if self.light :
            x = global_avg_pooling(x)

        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2) :
                x = fully_connected(x, channel, use_bias, scope='linear_' + str(i))
                x = relu(x)

            gamma = fully_connected(x, channel, use_bias, scope='gamma')
            beta = fully_connected(x, channel, use_bias, scope='beta')

            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, channel])
            beta = tf.reshape(beta, shape=[self.batch_size, 1, 1, channel])

            return gamma, beta

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        D_CAM_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            local_x, local_cam, local_heatmap = self.discriminator_local(x_init, reuse=reuse, scope='local')
            global_x, global_cam, global_heatmap, quant_loss, ppl = self.discriminator_global(
                x_init, reuse=reuse, scope='global')

            D_logit.extend([local_x, global_x])
            D_CAM_logit.extend([local_cam, global_cam])

            return D_logit, D_CAM_logit, local_heatmap, global_heatmap, quant_loss, ppl

    def discriminator_global(self, x_init, reuse=False, scope='discriminator_global'):
        with tf.variable_scope(scope, reuse=reuse):
            quant_loss = 0
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 1):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.2)
                if i in self.quant_layers:
                    diff, ppl = self.quantize[i](x, reuse, layer=i)
                    quant_loss += diff
                channel = channel * 2

            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel = channel * 2

            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))


            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap, quant_loss, ppl

    def discriminator_local(self, x_init, reuse=False, scope='discriminator_local'):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis - 2 - 1):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='conv_last')
            x = lrelu(x, 0.2)

            channel = channel * 2

            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)

            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = lrelu(x, 0.2)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='D_logit')

            return x, cam_logit, heatmap

    ##################################################################################
    # Model
    ##################################################################################

    def generate_a2b(self, x_A, reuse=False):
        out, cam, _ = self.generator(x_A, reuse=reuse, scope="generator_B")

        return out, cam

    def generate_b2a(self, x_B, reuse=False):
        out, cam, _ = self.generator(x_B, reuse=reuse, scope="generator_A")

        return out, cam

    def discriminate_real(self, x_A, x_B):
        real_A_logit, real_A_cam_logit, _, _, quant_loss_A, ppl_A = self.discriminator(x_A,
                                                                        scope="discriminator_A")
        real_B_logit, real_B_cam_logit, _, _, quant_loss_B, ppl_B = self.discriminator(x_B,
                                                                                       scope="discriminator_B")

        return real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit, \
               quant_loss_A+quant_loss_B, ppl_A+ppl_B

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit, fake_A_cam_logit, _, _, quant_loss_A, ppl_A = self.discriminator(x_ba, reuse=True,
                                                                   scope="discriminator_A")
        fake_B_logit, fake_B_cam_logit, _, _, quant_loss_B, ppl_B = self.discriminator(x_ab,
                                                                                       reuse=True,
                                                                   scope="discriminator_B")

        return fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit, \
               quant_loss_A+quant_loss_B, (ppl_A+ppl_B)/2

    def gradient_panalty(self, real, fake, scope="discriminator_A"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, cam_logit, _, _, _, _ = self.discriminator(interpolated, reuse=True,
                                                               scope=scope)


        GP = []
        cam_GP = []

        for i in range(2) :
            grad = tf.gradients(logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        for i in range(2) :
            grad = tf.gradients(cam_logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))


        return sum(GP), sum(cam_GP)

    def build_model(self):
        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')


            """ Input Image"""
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)


            #gpu_device = '/gpu:0'
            #trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))
            #trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))
            trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True))
            trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True))


            trainA_iterator = trainA.make_one_shot_iterator()
            trainB_iterator = trainB.make_one_shot_iterator()

            self.domain_A = trainA_iterator.get_next()
            self.domain_B = trainB_iterator.get_next()

            """ Define Generator, Discriminator """
            x_ab, cam_ab = self.generate_a2b(self.domain_A) # real a
            x_ba, cam_ba = self.generate_b2a(self.domain_B) # real b

            x_aba, _ = self.generate_b2a(x_ab, reuse=True) # real b
            x_bab, _ = self.generate_a2b(x_ba, reuse=True) # real a

            x_aa, cam_aa = self.generate_b2a(self.domain_A, reuse=True) # fake b
            x_bb, cam_bb = self.generate_a2b(self.domain_B, reuse=True) # fake a

            real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit, real_quant_loss,\
            real_ppl = self.discriminate_real(self.domain_A, self.domain_B)
            fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit, fake_quant_loss,  \
            fake_ppl = self.discriminate_fake(x_ba, x_ab)
            self.ppl = real_ppl + fake_ppl

            """ Define Loss """
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP_A, GP_CAM_A = self.gradient_panalty(real=self.domain_A, fake=x_ba, scope="discriminator_A")
                GP_B, GP_CAM_B = self.gradient_panalty(real=self.domain_B, fake=x_ab, scope="discriminator_B")
            else :
                GP_A, GP_CAM_A  = 0, 0
                GP_B, GP_CAM_B = 0, 0

            G_ad_loss_A = (generator_loss(self.gan_type, fake_A_logit) + generator_loss(self.gan_type, fake_A_cam_logit))
            G_ad_loss_B = (generator_loss(self.gan_type, fake_B_logit) + generator_loss(self.gan_type, fake_B_cam_logit))

            D_ad_loss_A = (discriminator_loss(self.gan_type, real_A_logit, fake_A_logit) + discriminator_loss(self.gan_type, real_A_cam_logit, fake_A_cam_logit) + GP_A + GP_CAM_A)
            D_ad_loss_B = (discriminator_loss(self.gan_type, real_B_logit, fake_B_logit) + discriminator_loss(self.gan_type, real_B_cam_logit, fake_B_cam_logit) + GP_B + GP_CAM_B)

            reconstruction_A = L1_loss(x_aba, self.domain_A) # reconstruction
            reconstruction_B = L1_loss(x_bab, self.domain_B) # reconstruction

            identity_A = L1_loss(x_aa, self.domain_A)
            identity_B = L1_loss(x_bb, self.domain_B)

            cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
            cam_B = cam_loss(source=cam_ab, non_source=cam_bb)

            Generator_A_gan = self.adv_weight * G_ad_loss_A
            Generator_A_cycle = self.cycle_weight * reconstruction_B
            Generator_A_identity = self.identity_weight * identity_A
            Generator_A_cam = self.cam_weight * cam_A

            Generator_B_gan = self.adv_weight * G_ad_loss_B
            Generator_B_cycle = self.cycle_weight * reconstruction_A
            Generator_B_identity = self.identity_weight * identity_B
            Generator_B_cam = self.cam_weight * cam_B


            Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + Generator_A_cam
            Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + Generator_B_cam


            Discriminator_A_loss = self.adv_weight * D_ad_loss_A
            Discriminator_B_loss = self.adv_weight * D_ad_loss_B

            self.Generator_loss = Generator_A_loss + Generator_B_loss + regularization_loss(
                'generator') + fake_quant_loss
            self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + \
                                      regularization_loss('discriminator') + real_quant_loss + fake_quant_loss


            """ Result Image """
            self.fake_A = x_ba
            self.fake_B = x_ab

            self.real_A = self.domain_A
            self.real_B = self.domain_B


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)


            """" Summary """
            self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
            self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

            self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
            self.G_A_gan = tf.summary.scalar("G_A_gan", Generator_A_gan)
            self.G_A_cycle = tf.summary.scalar("G_A_cycle", Generator_A_cycle)
            self.G_A_identity = tf.summary.scalar("G_A_identity", Generator_A_identity)
            self.G_A_cam = tf.summary.scalar("G_A_cam", Generator_A_cam)

            self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
            self.G_B_gan = tf.summary.scalar("G_B_gan", Generator_B_gan)
            self.G_B_cycle = tf.summary.scalar("G_B_cycle", Generator_B_cycle)
            self.G_B_identity = tf.summary.scalar("G_B_identity", Generator_B_identity)
            self.G_B_cam = tf.summary.scalar("G_B_cam", Generator_B_cam)

            self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
            self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.scalar(var.name + "_min", tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_max", tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_mean", tf.reduce_mean(var)))

            g_summary_list = [self.G_A_loss, self.G_A_gan, self.G_A_cycle, self.G_A_identity, self.G_A_cam,
                              self.G_B_loss, self.G_B_gan, self.G_B_cycle, self.G_B_identity, self.G_B_cam,
                              self.all_G_loss]

            g_summary_list.extend(self.rho_var)
            d_summary_list = [self.D_A_loss, self.D_B_loss, self.all_D_loss]

            self.G_loss = tf.summary.merge(g_summary_list)
            self.D_loss = tf.summary.merge(d_summary_list)
            # self.ppl = tf.summary.scalar('Perplexity', self.ppl)
            if self.test_train:
                """ Test """
                self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
                self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')

                self.test_fake_B, _ = self.generate_a2b(self.test_domain_A, reuse=True)
                self.test_fake_A, _ = self.generate_b2a(self.test_domain_B, reuse=True)
        elif self.phase == 'test':
            self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')

            self.test_fake_B, _ = self.generate_a2b(self.test_domain_A)
            self.test_fake_A, _ = self.generate_b2a(self.test_domain_B)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr

        for epoch in range(start_epoch, self.epoch):
            # lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            if self.decay_flag :
                #lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
            for idx in range(start_batch_id, self.iteration):
                #print("print_idx={%d}" % (idx))
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str, ppl = self.sess.run([self.D_optim,
                                                        self.Discriminator_loss, self.D_loss,
                                                             self.ppl], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                g_loss = None
                if (counter - 1) % self.n_critic == 0 :
                    batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B,
                                                                                                            self.fake_A, self.fake_B,
                                                                                                            self.G_optim,
                                                                                                            self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                if idx % 1000==0:
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, ppl: %.4f"
                          "" % (epoch, idx, self.iteration, time.time() - start_time, d_loss,
                                g_loss, ppl))

                if np.mod(idx+1, self.print_freq) == 0 :
                    # print("save images")#change
                    # print(self.sample_dir)
                    save_images(batch_A_images, [self.batch_size, 1],
                                '{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                '{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                # if np.mod(idx + 1, self.save_freq) == 0:
                #     self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # if epoch % 2 == 0:
            self.test(epoch)
            # save model for final step
            if np.mod(epoch+1, 5) == 0:
                self.save(self.checkpoint_dir, counter)




    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_solid = False
        while not save_solid:
            try:
                self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)
                # print('ckpt saved...')
                save_solid = True
            except:
                pass

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self, epoch):
        if not self.test_train:
            tf.global_variables_initializer().run()
            self.saver = tf.train.Saver()
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load :
                print(" [*] Load SUCCESS")
            else :
                print(" [!] Load failed...")

        test_A_root = '{}'.format(self.dataset_name+'/testA')
        test_B_root = '{}'.format(self.dataset_name+'/testB')
        train_A_root = '{}'.format(self.dataset_name + '/trainA')
        train_B_root = '{}'.format(self.dataset_name + '/trainB')
        test_A_files = glob('{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('{}/*.*'.format(self.dataset_name + '/testB'))
        A2B_root = os.path.join(self.result_dir, '{:03d}-{}'.format(epoch, 'A-B'))
        B2A_root = os.path.join(self.result_dir, '{:03d}-{}'.format(epoch, 'B-A'))
        # check_folder(self.result_dir)
        check_folder(A2B_root)
        check_folder(B2A_root)

        for sample_file  in test_A_files : # A -> B
            # print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))

            image_path = os.path.join(A2B_root, os.path.basename(sample_file))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

        for sample_file  in test_B_files : # B -> A
            
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))

            image_path = os.path.join(B2A_root, os.path.basename(sample_file))

            fake_img = self.sess.run(self.test_fake_A, feed_dict = {self.test_domain_B : sample_image})

            save_images(fake_img, [1, 1], image_path)

