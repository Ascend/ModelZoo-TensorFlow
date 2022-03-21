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
import numpy as np
import time
import os
import cv2
import npu_bridge
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import FixedLossScaleManager

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    checkimage,
    imsave,
    imread,
    load_data,
    preprocess,
    modcrop,
    make_sub_data,
)

from PSNR import psnr, psnr_ycbcr


class ESPCN(object):
    def __init__(self,
                 sess,
                 image_size,
                 is_train,
                 scale,
                 batch_size,
                 c_dim,
                 test_img,
                 data_dir,
                 logger,
                 ):

        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.test_img = test_img
        self.data_dir = data_dir
        self.logger = logger

    def build_model(self, config, input_i=None):
        """
        to init model
        """
        if self.is_train:
            self.images = tf.placeholder(
                tf.float32, [config.batch_size, self.image_size, self.image_size, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [config.batch_size, self.image_size * self.scale, self.image_size *
                                                      self.scale, self.c_dim], name='labels')
        else:
            '''
                Because the test need to put image to model,
                so here we don't need do preprocess, so we set input as the same with preprocess output
            '''
            if self.c_dim == 1:
                print(input_i.shape)
                self.h, self.w = input_i.shape
            else:
                self.h, self.w, _ = input_i.shape
            self.images = tf.placeholder(
                tf.float32, [1, self.h, self.w, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [
                                         1, self.h * self.scale, self.w * self.scale, self.c_dim], name='labels')

        self.weights = {
            'w1': tf.Variable(tf.random.normal([5, 5, self.c_dim, 64], stddev=np.sqrt(2.0 / 25 / 3)), name='w1'),
            'w2': tf.Variable(tf.random.normal([3, 3, 64, 32], stddev=np.sqrt(2.0 / 9 / 64)), name='w2'),
            'w3': tf.Variable(tf.random.normal([3, 3, 32, self.c_dim * self.scale * self.scale],
                                               stddev=np.sqrt(2.0 / 9 / 32)), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale], name='b3'))
        }

        self.pred = self.model()

        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver()  # To save checkpoint

    def model(self):
        """
        model structure
        """
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[
                           1, 1, 1, 1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[
                           1, 1, 1, 1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[
                             1, 1, 1, 1], padding='SAME') + self.biases['b3']  # This layer don't need ReLU

        ps = self.PS(conv3, self.scale)
        return tf.nn.tanh(ps, name="output")

    def _phase_shift(self, I, r):
        """
        NOTE: train with batch size
        one channel
        Helper function with main phase shift operation
        """
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a * r, b * r, 1))

    def _phase_shift_test(self, I, r):
        """
        NOTE:test without batchsize
        """
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))  # bsize = 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))

    def PS(self, X, r):
        """
        PixelShuffle
        Main OP that you can arbitrarily use in you tensorflow code
        X [bsize, height, width, channel]
        """
        if self.c_dim == 3:
            Xc = tf.split(X, 3, 3)  # tf.split second:nums third:shape[3]
        else:
            Xc = tf.split(X, 1, 3)

        if self.is_train:
            if self.c_dim == 3:
                X = tf.concat([self._phase_shift(x, r)
                               for x in Xc], 3)  # Do the concat RGB
            else:
                X = [self._phase_shift(x, r) for x in Xc] # Do the concat y
        else:
            if self.c_dim == 3:
                X = tf.concat([self._phase_shift_test(x, r)
                               for x in Xc], 3)  # Do the concat RGB
            else:
                X = [self._phase_shift_test(x, r) for x in Xc]  # Do the concat y
        return X

    def test(self, config, logger, index):
        """
        test: Make sub_input and sub_label, if is_train false more return nx, ny
        """
        data = load_data(config)
        input_, label_ = make_sub_data(data, config)
        input_i = input_[index]
        label_i = label_[index]
        self.build_model(config, input_i)
        self.load(config.checkpoint_dir, config.data_dir)
        print(config.is_train)

        logger.info("Now Start Testing...")

        result = self.pred.eval(
            {self.images: input_i.reshape(1, self.h, self.w, self.c_dim)})
        output_i = np.squeeze(result)

        lr = modcrop(label_i) * 255.
        hr = modcrop(output_i) * 255.

        br = cv2.resize(input_i, None, fx=config.scale, fy=config.scale, interpolation=cv2.INTER_CUBIC)
        br = modcrop(br)*255.

        # br =  np.array(br, dtype='uint8')
        # lr =  np.array(lr, dtype='uint8')
        # hr =  np.array(hr, dtype='uint8')
        # cv2.imwrite("br.bmp",br)
        # cv2.imwrite("lr.bmp", lr)
        # cv2.imwrite("hr.bmp", hr)

        logger.info("shape:{},{},{}".format(lr.shape, hr.shape, br.shape))
        if self.c_dim == 1:
            bi = psnr(lr, br, scale=3)
            temp = psnr(lr, hr, scale=3)
        else:
            bi = 0
            temp = psnr_ycbcr(lr, hr, scale=3)
        logger.info("{},{}".format(bi, temp))
        return temp, bi

    def train(self, config, logger):
        """
        start to train
        """
        self.build_model(config)
        savepath = input_setup(config)                      #add 性能看护需要此绝对路径savepath，否则无法读取到train.h5
        data_dir = checkpoint_dir(config,savepath)
        #("data_dir",data_dir)
        logger.info("data_dir:{}".format(data_dir))
        input_, label_ = read_data(data_dir)

        logger.info("input_.shape:{}, label_.shape:{}".format(input_.shape, label_.shape))
        logger.info("config.is_train:{}".format(config.is_train))

        # Stochastic gradient descent with the standard backpropagation
        # self.train_op = tf.train.AdamOptimizer(
        #     learning_rate=config.learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate)
        opt_tmp = self.train_op
        # loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(loss_scale = 1)
        loss_scale_manager = FixedLossScaleManager(loss_scale=1)
        self.train_op = NPULossScaleOptimizer(opt_tmp, loss_scale_manager).minimize(self.loss)

        # tf.initialize_all_variables().run()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        counter = 0
        perf_list=[]
        fps_list=[]

        # self.load(config.checkpoint_dir,config.data_dir)
        # Train
        min_loss = float("inf")
        if config.is_train:
            logger.info("Now Start Training...")
            for ep in range(config.epoch):  #30
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size   #18
                total_loss = 0
                for idx in range(0, batch_idxs):
                    time_ = time.time()
                    batch_images = input_[
                        idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = label_[
                        idx * config.batch_size: (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={
                                           self.images: batch_images, self.labels: batch_labels})
                    total_loss += err
                    if counter % 10 == 0:
                        perf = (time.time() - time_) / 10
                        perf_list.append(perf)
                        perf_mean = np.mean(perf_list)
                        fps = config.batch_size / perf
                        fps_list.append(fps)
                        fps_mean = np.mean(fps_list)
                        logger.info("Epoch: [%2d], step: [%2d], time: %4.4f fps: %4.4f  loss: %.8f perf_mean: %4.4f fps_mean: %4.4f" % (
                            (ep + 1), counter, perf *1000, fps, err, perf_mean*1000, fps_mean))
                    if counter % 500 == 0:      #30*18=540
                        if total_loss < min_loss:
                            logger.info("Epoch: [%2d], time: [%4.4f], total_loss: [%.8f]" % (
                                (ep + 1), time.time() - time_, total_loss))
                            min_loss = total_loss
                            self.save(config.checkpoint_dir, counter)

    def load(self, checkpoint_dir, data_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        # give the model name by label_size
        model_dir = "%s_imgsize%s_scale%s_channel%s" % ("espcn", self.image_size, self.scale, self.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        # Check the checkpoint is exist
        if ckpt and ckpt.model_checkpoint_path:
            # convert the unicode to string
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.logger.info(os.path.join(data_dir, ckpt_path))
            self.saver.restore(self.sess, os.path.join(data_dir, ckpt_path))
            self.logger.info("\n Checkpoint Loading Success! %s\n\n" % ckpt_path)
        else:
            self.logger.info("\n! Checkpoint Loading Failed \n\n")

    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "ESPCN.model"
        model_dir = "%s_imgsize%s_scale%s_channel%s" % ("espcn", self.image_size, self.scale, self.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)  

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name))
