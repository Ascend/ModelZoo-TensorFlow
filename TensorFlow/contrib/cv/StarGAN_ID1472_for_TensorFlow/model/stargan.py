# -*- coding:utf-8 -*-
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


import os
import logging
import datetime
import time
import tensorflow as tf
import numpy as np
from model.models import network_Gen, network_Dis
import sys

sys.path.append("../data")
from data import data_loader


class StarGAN(object):
    '''
    Load CelebA dataset.
    Train and test StarGAN.
    '''
    def __init__(self, sess, tf_flags):
        '''
        
        :param sess: 
        :param tf_flags: 
        '''
        self.sess = sess
        self.dtype = tf.float32

        # Directories.
        self.output_dir = tf_flags.output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.checkpoint_prefix = "model"
        self.saver_name = "checkpoint"
        self.summary_dir = os.path.join(self.output_dir, "summary")
        self.sample_dir = os.path.join(self.output_dir, "sample")
        self.result_dir = os.path.join(self.output_dir, "results")

        # model configuration.
        self.is_training = (tf_flags.phase == "train")  # train or test.
        self.image_h = tf_flags.image_size
        self.image_w = tf_flags.image_size
        assert self.image_h == self.image_w
        self.g_conv_dim = tf_flags.g_conv_dim
        self.d_conv_dim = tf_flags.d_conv_dim
        self.g_repeat_num = tf_flags.g_repeat_num
        self.d_repeat_num = tf_flags.d_repeat_num
        self.lambda_cls = tf_flags.lambda_cls
        self.lambda_rec = tf_flags.lambda_rec
        self.lambda_gp = tf_flags.lambda_gp

        # training configuration
        self.batch_size = tf_flags.batch_size
        self.image_c = 3
        self.c_dim = tf_flags.c_dim
        self.selected_attrs = tf_flags.selected_attrs.split(" ")
        assert len(self.selected_attrs) == self.c_dim
        self.d_train_repeat = tf_flags.d_train_repeat
        self.init_learning_rate = tf_flags.init_learning_rate
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        # The learning_rate decay parameters
        self.lr_update_step = tf_flags.lr_update_step
        self.num_step_decay = tf_flags.num_step_decay
        self.beta1 = tf_flags.beta1
        self.beta2 = tf_flags.beta2
        # for saving test results
        self.test_concatenate = (tf_flags.test_concatenate == "true")

        # placeholder. images and attribute labels.
        # attribute labels contain a real_c, a label_trg.
        # c is condition, the dimension of condition is c_dim, default is 5.
        # Here is only define placeholder, the true data is at the process of training.
        self.x_real = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c], name="x_real")
        self.label_org = tf.placeholder(self.dtype, [None, self.c_dim], name="label_org")
        self.c_org = tf.placeholder(self.dtype, [None, self.c_dim], name="c_org")
        self.label_trg = tf.placeholder(self.dtype, [None, self.c_dim], name="label_trg")
        self.c_trg = tf.placeholder(self.dtype, [None, self.c_dim], name="c_trg")
        # x_fixed c_fixed
        self.x_fixed = tf.placeholder(self.dtype, [None, self.image_h, self.image_w, self.image_c], name="x_fixed")
        self.c_fixed = tf.placeholder(self.dtype, [None, self.c_dim], name="c_fixed")

        # train
        if self.is_training:
            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            log_file = os.path.join(self.output_dir, "StarGAN.log")
            logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                                filename=log_file,
                                level=logging.DEBUG,
                                filemode='a+')
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self._build_test()

    def label2onehot(self, labels, dim):
        '''
        
        :param labels: 
        :param dim: 
        :return: 
        '''
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = np.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        '''
        
        :param c_org: 
        :param c_dim: 
        :param dataset: 
        :param selected_attrs: 
        :return: 
        '''
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.copy()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(np.ones(np.array(c_org).shape(0)) * i, c_dim)

            c_trg_list.append(c_trg)

        return c_trg_list

    def denorm(self, x):
        '''
        
        :param x: 
        :return: 
        '''
        """Convert the range from [-1, 1] to [0, 1]."""

        image = (((x + 1.) / 2).clip(0., 1.)) * 255

        return image

    def _build_training(self):
        '''
        
        :return: 
        '''
        # Generator network.
        self.fake_images = network_Gen(name="G", in_data=self.x_real, c=self.c_trg, num_filters=self.g_conv_dim,
                                       g_n_blocks=self.g_repeat_num, reuse=False)
        self.rec_images = network_Gen(name="G", in_data=self.fake_images, c=self.c_trg, num_filters=self.g_conv_dim,
                                      g_n_blocks=self.g_repeat_num, reuse=True)
        # Generator network Variables.
        self.G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")

        self.fixed_fake_images = network_Gen(name="G", in_data=self.x_fixed, c=self.c_fixed,
                                             num_filters=self.g_conv_dim, g_n_blocks=self.g_repeat_num, reuse=True)

        # Discriminator network.
        self.pre_real_src, self.pre_real_cls = network_Dis(name="D", in_data=self.x_real, image_size=self.image_h,
                                                           num_filters=self.d_conv_dim, c_dim=self.c_dim,
                                                           d_n_blocks=self.d_repeat_num, reuse=False)
        self.pre_fake_src, self.pre_fake_cls = network_Dis(name="D", in_data=self.fake_images, image_size=self.image_h,
                                                           num_filters=self.d_conv_dim, c_dim=self.c_dim,
                                                           d_n_blocks=self.d_repeat_num, reuse=True)
        # Compute for gradient penalty.
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = alpha * self.x_real + (1 - alpha) * self.fake_images
        pre_inter_src, _ = network_Dis(name="D", in_data=interpolated, image_size=self.image_h,
                                       num_filters=self.d_conv_dim, c_dim=self.c_dim, d_n_blocks=self.d_repeat_num,
                                       reuse=True)
        # Discriminator network Variables.
        self.D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")

        # loss

        # Discriminator loss
        # D real loss.
        self.D_real_loss = -tf.reduce_mean(self.pre_real_src)
        # D cls loss
        self.D_cls_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_org, logits=self.pre_real_cls))
        # D fake loss
        self.D_fake_loss = tf.reduce_mean(self.pre_fake_src)
        # D gp loss
        grad = tf.gradients(pre_inter_src, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)  # l2 norm
        self.D_loss_gp = tf.reduce_mean(tf.reduce_mean(tf.square(grad_norm - 1.)))

        # D loss
        self.D_loss = self.D_real_loss + self.D_fake_loss + self.lambda_cls * self.D_cls_loss + self.lambda_gp * self.D_loss_gp

        # Generator loss
        # G fake loss
        self.G_fake_loss = -tf.reduce_mean(self.pre_fake_src)
        # G cls loss
        self.G_cls_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_trg, logits=self.pre_fake_cls))
        # G rec loss, input is fake images, output is rec images.
        self.G_rec_loss = tf.reduce_mean(tf.abs(self.x_real - self.rec_images))

        # G loss
        self.G_loss = self.G_fake_loss + self.lambda_rec * self.G_rec_loss + self.lambda_cls * self.G_cls_loss

        # optimizer
        start_time = time.time()
        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2).minimize(
            self.G_loss, var_list=self.G_variables, name="G_opt")
        self.D_opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=self.beta2).minimize(
            self.D_loss, var_list=self.D_variables, name="D_opt")

        # summary
        tf.summary.scalar('D/loss_real', self.D_real_loss)
        tf.summary.scalar('D/loss_fake', self.D_fake_loss)
        tf.summary.scalar('D/loss_cls', self.D_cls_loss)
        tf.summary.scalar('D/loss_gp', self.D_loss_gp)
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G/loss_fake', self.G_fake_loss)
        tf.summary.scalar('G/loss_rec', self.G_rec_loss)
        tf.summary.scalar('G/loss_cls', self.G_cls_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('Perf', time.time()-start_time)

        self.summary = tf.summary.merge_all()
        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()

    def train(self, image_root, metadata_path, training_steps, summary_steps, checkpoint_steps, save_steps):
        '''
        
        :param image_root: 
        :param metadata_path: 
        :param training_steps: 
        :param summary_steps: 
        :param checkpoint_steps: 
        :param save_steps: 
        :return: 
        '''
        step_num = 0
        # restore last checkpoint
        latest_checkpoint = tf.train.latest_checkpoint("model_output_20180303172402/checkpoint")
        # use pretrained model, it can be self.checkpoint_dir, "", or you can appoint the saved checkpoint path.
        # e.g., model_output_20180303114343/checkpoint

        if latest_checkpoint:
            step_num = int(os.path.basename(latest_checkpoint).split("-")[1])
            assert step_num > 0, "Please ensure checkpoint format is model-*.*."
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("{}: Resume training from step {}. Loaded checkpoint {}".format(datetime.datetime.now(),
                                                                                         step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer())  # init all variables
            logging.info("{}: Init new training".format(datetime.datetime.now()))

        # data
        # define a object of CelebADataset.
        dataset = data_loader.CelebADataset(image_root=image_root, metadata_path=metadata_path,
                                            is_training=self.is_training, batch_size=self.batch_size,
                                            image_h=self.image_h, image_w=self.image_w,
                                            image_c=self.image_c)
        data_generate = dataset.batch_generator_numpy()
        # print(type(data_generate))

        # x_fixed
        data_gen_ = next(data_generate)
        x_fixed = data_gen_["images"]
        c_fixed = data_gen_["attribute"]
        c_fixed_list = self.create_labels(c_fixed, self.c_dim, 'CelebA', self.selected_attrs)

        # train
        c_time = time.time()
        lr = self.init_learning_rate
        for c_step in range(step_num + 1, training_steps + 1):
            # Decay learning rates.
            if c_step % self.lr_update_step == 0 and c_step > (training_steps - self.num_step_decay):
                lr -= (self.init_learning_rate / float(self.num_step_decay))  # linear decay
                print('Decayed learning rates, lr: {}.'.format(lr))

            # # alpha, a uniform distribution tensor, shape is [batch_size, h, w, c].
            # alpha = np.random.rand(self.batch_size, 1, 1, 1)

            # minval is 0, maxval is 1. There, must be numpy ndarray.

            data_gen = next(data_generate)
            rand_index = np.arange(self.batch_size)
            # not fixed order
            np.random.shuffle(rand_index)
            label_trg = data_gen["attribute"][rand_index]

            c_feed_dict = {
                # numpy ndarray
                self.x_real: data_gen["images"],
                self.label_org: data_gen["attribute"],
                self.c_org: data_gen["attribute"],
                self.label_trg: label_trg,
                self.c_trg: label_trg,
                # self.alpha: alpha,
                self.lr: lr
            }

            # Refer to Pytorch StarGAN, train D network d_train_repeat times, then train G network one time.
            '''
            # G and D are training at the same time.
            self.ops = [self.G_opt, self.D_opt]
            self.sess.run(self.ops, feed_dict=c_feed_dict)
            '''
            # train D network d_train_repeat times, then train G network one time.
            # train D.
            self.sess.run(self.D_opt, feed_dict=c_feed_dict)
            # train G
            if c_step % self.d_train_repeat == 0:
                self.sess.run(self.G_opt, feed_dict=c_feed_dict)

            # save summary
            if c_step % summary_steps == 0:
                c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                self.writer.add_summary(c_summary, c_step)

                e_time = time.time() - c_time
                e_time = str(datetime.timedelta(seconds=e_time))[:-7]
                log_loss_info = "Elapsed [{}], Iteration [{}/{}]{}".format(e_time, c_step, training_steps,
                                                                           self._print_summary(c_summary))
                print(log_loss_info)

                c_time = time.time()  # update time

            # save checkpoint
            if c_step % checkpoint_steps == 0:
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                                global_step=c_step)
                logging.info("{}: Iteration_{} Saved checkpoint".format(datetime.datetime.now(), c_step))

            # save training images
            if c_step % save_steps == 0:
                x_fake_list = [x_fixed]
                # print(x_fixed.shape)
                for c_fixed in c_fixed_list:
                    fixed_feed_dict = {
                        # numpy ndarray
                        self.x_fixed: x_fixed,
                        self.c_fixed: c_fixed,
                    }
                    fixed_fake_images = self.sess.run(self.fixed_fake_images, feed_dict=fixed_feed_dict)
                    # print(np.array(fixed_fake_images).shape)
                    x_fake_list.append(fixed_fake_images)
                x_concat = np.concatenate(x_fake_list, axis=2)
                # print(x_concat.shape)

                # sample_path = os.path.join(self.sample_dir, '{}-images.png'.format(c_step + 1))
                # data_loader.save_images(self.denorm(x_concat), sample_path)

                data_loader.save_images(x_concat, '{}/train_{}_{:06d}.png'.format(self.sample_dir, "stargan", c_step))

        logging.info("{}: Done training".format(datetime.datetime.now()))

    def _build_test(self):
        '''
        
        :return: 
        '''
        # Generator network. network_Gen() is Generator network.
        # input real images and label_trg, output the fake images. fake images is our target, e.g. Aged image.
        self.fake_images = network_Gen(name="G", in_data=self.x_real, c=self.label_trg, num_filters=self.g_conv_dim,
                                       g_n_blocks=self.g_repeat_num, reuse=False)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        # define saver, after the network!

    def load(self, checkpoint_name=None):
        '''
        
        :param checkpoint_name: 
        :return: 
        '''
        # restore checkpoint
        print("{}: Loading checkpoint...".format(datetime.datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(" loaded {}".format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(" loaded {}".format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    "No checkpoints found in {}".format(self.checkpoint_dir))

    def test(self, image_root, metadata_path):
        '''
        
        :param image_root: 
        :param metadata_path: 
        :return: 
        '''
        # In the phase
        # 1) you can use the 2000 images(see data/data_loader.py for detail,
        # the number of test images is 2000) to generate one mode's generated images.
        # 2) can refer to Pytorch code, solver.py.

        # for results
        import shutil
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        if self.test_concatenate:
            os.makedirs(os.path.join(self.result_dir, "src_images"))
            os.makedirs(os.path.join(self.result_dir, "generates"))
        else:
            os.makedirs(os.path.join(self.result_dir, "src"))
            for c_ in range(1, self.c_dim + 1):
                os.makedirs(os.path.join(self.result_dir, "generate{}".format(c_)))

        # data
        # define a object of CelebADataset.
        dataset_test = data_loader.CelebADataset(image_root=image_root, metadata_path=metadata_path,
                                                 is_training=False, batch_size=self.batch_size,
                                                 image_h=self.image_h, image_w=self.image_w,
                                                 image_c=self.image_c)
        data_generate_test = dataset_test.batch_generator_numpy()

        from tqdm import tqdm
        for i in tqdm(range(2000)):
            data_gen_test = next(data_generate_test)
            x_real = data_gen_test["images"]
            c_org = data_gen_test["attribute"]
            c_trg_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)

            x_fake_list = [x_real]
            for c_trg in c_trg_list:
                c_feed_dict = {
                    # numpy ndarray
                    self.x_real: x_real,
                    self.label_trg: c_trg,
                }
                fake_images = self.sess.run(self.fake_images, feed_dict=c_feed_dict)
                x_fake_list.append(fake_images)

            if self.test_concatenate:
                # # save images with concatenate
                # x_concat = np.concatenate(x_fake_list, axis=2)
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                # data_loader.save_images(x_concat, result_path, self.batch_size)

                # save src images
                src_images = [x_real for _ in range(self.c_dim)]

                x_concat = np.concatenate(src_images, axis=2)
                result_path = os.path.join(self.result_dir, 'src_images/{}-images.jpg'.format(i + 1))
                data_loader.save_images(x_concat, result_path, self.batch_size)

                # save generates images
                x_concat = np.concatenate(x_fake_list[1:], axis=2)
                result_path = os.path.join(self.result_dir, 'generates/{}-images.jpg'.format(i + 1))
                data_loader.save_images(x_concat, result_path, self.batch_size)
            else:
                # save image only
                x_concat = x_fake_list[0]
                result_path = os.path.join(self.result_dir, 'src/{}-images.jpg'.format(i + 1))
                data_loader.save_images(x_concat, result_path, self.batch_size)
                # Black_Hair Blond_Hair Brown_Hair Male Young
                for c_ in range(1, self.c_dim + 1):
                    x_concat = x_fake_list[c_]
                    result_path = os.path.join(self.result_dir, 'generate{}/{}-images.jpg'.format(c_, i + 1))
                    data_loader.save_images(x_concat, result_path, self.batch_size)

    def _make_aux_dirs(self):
        '''
        
        :return: 
        '''
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        '''
        
        :param summary_string: 
        :return: 
        '''
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append(", {}: {:.4f}".format(val.tag, val.simple_value))
        return " ".join(result)
