# Deep Sparse Representation-based Classification
# https://arxiv.org/abs/1904.11093
# Mahdi Abavisani
# mahdi.abavisani@rutgers.edu
# Built upon https://github.com/panji1990/Deep-subspace-clustering-networks
#        and https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks
#
# Citation:  M. Abavisani and V. M. Patel, "Deep sparse representation-based clas- sification,"
#            IEEE Signal Processing Letters, vol. 26, no. 6, pp. 948-952, June 2019.
#            DOI:10.1109/LSP.2019.2913022
#
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
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
import argparse
import time
import cv2
import npu_bridge
import random
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


class ConvAE(object):
    def __init__(self,train_data,test_data, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, batch_size=200, train_size=100,reg=None, \
                 denoise=False, model_path=None, restore_path=None, \
                 logs_path='./logs'):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = batch_size - train_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0

        tf.set_random_seed(2019)
        weights = self._initialize_weights()

        self.train = tf.convert_to_tensor(train_data)
        self.test = tf.convert_to_tensor(test_data)
        self.learning_rate = tf.constant(1e-4)

        self.x = tf.concat([self.train, self.test], axis=0)  # Concat testing and training samples

        latent, latents, shape = self.encoder(self.x, weights)
        latent_shape = tf.shape(latent)

        # Slice the latent space features to separate training and testing latent features
        latent_train = tf.slice(latent, [0, 0, 0, 0],
                                [self.train_size, latent_shape[1], latent_shape[2], latent_shape[3]])
        latent_test = tf.slice(latent, [self.train_size, 0, 0, 0],
                               [self.test_size, latent_shape[1], latent_shape[2], latent_shape[3]])

        # Vectorize the features
        z_train = tf.reshape(latent_train, [self.train_size, -1])
        z_test = tf.reshape(latent_test, [self.test_size, -1])
        z = tf.reshape(latent, [self.batch_size, -1])

        Coef = weights['Coef']  # This is \theta in the paper

        z_test_c = tf.matmul(Coef, z_train)
        z_c = tf.concat([z_train, z_test_c], axis=0)
        latent_c_test = tf.reshape(z_test_c, tf.shape(latent_test))

        latent_c_pretrain = tf.concat([latent_train, latent_test], axis=0)  # used in pretraining stage
        latent_c = tf.concat([latent_train, latent_c_test], axis=0)  # used in the main model

        self.x_r_pretrain = self.decoder(latent_c_pretrain, weights, shape)  # used in pretraining stage
        self.x_r = self.decoder(latent_c, weights, shape)  # used in the main model

        self.Coef_test = Coef

        self.AE = tf.concat([z_train, z_test], axis=0)  # Autoencoder features to be used in benchmarks comparison

        # l_2 reconstruction loss

        self.loss_pretrain = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.x_r_pretrain), 2.0))

        self.reconst_cost_x = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.x_r), 2.0))
        tf.summary.scalar("recons_loss", self.reconst_cost_x)

        self.reg_losses = tf.reduce_sum(tf.pow(Coef, 2.0))
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z_c, z), 2.0))

        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses)

        # TOTAL LOSS
        self.loss = self.reconst_cost_x + reg_constant1 * self.reg_losses + 0.5 * re_constant2 * self.selfexpress_losses

        self.merged_summary_op = tf.summary.merge_all()
        
        
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
#             self.loss)  # GradientDescentOptimizer #AdamOptimizer
#         self.optimizer_pretrain = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
#             self.loss_pretrain)  # GradientDescentOptimizer #AdamOptimizer

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=1024, incr_every_n_steps=100, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        self.loss_scale_optimizer = NPULossScaleOptimizer(self.optimizer, self.loss_scale_manager)
        self.train_step = self.loss_scale_optimizer.minimize(self.loss)
        
        self.optimizer_pretrain = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss_scale_manager_pretrain = ExponentialUpdateLossScaleManager(init_loss_scale=1024, incr_every_n_steps=100, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        self.loss_scale_optimizer_pretrains = NPULossScaleOptimizer(self.optimizer_pretrain, self.loss_scale_manager_pretrain)
        self.train_step_pretrain = self.loss_scale_optimizer_pretrains.minimize(self.loss_pretrain)
        
        
        self.init = tf.global_variables_initializer()
        tfconfig = tf.ConfigProto()
        custom_op = tfconfig.graph_options.rewrite_options.custom_optimizers.add()
        tfconfig.gpu_options.allow_growth = True
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"./","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"resnet_model/conv2d/Conv2Dresnet_model/batch_normalization/FusedBatchNormV3_Reduce","bp_point":"gradients/AddN_70","aic_metrics":"PipeUtilization"}')
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/ma-user/modelarts/outputs/train_url_0/","task_trace":"on"}')
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")
        tfconfig.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        tfconfig.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        
        
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(self.init)
        self.saver = tf.train.Saver(
            [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])  # to save the pretrained model
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        '''
        initializes weights for the model and soters them in a dictionary.
        '''

        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                       self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['enc1_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['enc_w1'] = tf.get_variable("enc_w1",
                                                shape=[self.kernel_size[1], self.kernel_size[1],
                                                       self.n_hidden[0],
                                                       self.n_hidden[1]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['enc_w2'] = tf.get_variable("enc_w2",
                                                shape=[self.kernel_size[2], self.kernel_size[2],
                                                       self.n_hidden[1],
                                                       self.n_hidden[2]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

        all_weights['dec_w0'] = tf.get_variable("dec1_w0",
                                                shape=[self.kernel_size[2], self.kernel_size[2],
                                                       self.n_hidden[1],
                                                       self.n_hidden[3]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

        all_weights['dec_w1'] = tf.get_variable("dec1_w1",
                                                shape=[self.kernel_size[1], self.kernel_size[1],
                                                       self.n_hidden[0],
                                                       self.n_hidden[1]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['dec_w2'] = tf.get_variable("dec1_w2",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                       self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))

        all_weights['enc_w3'] = tf.get_variable("enc_w3",
                                                shape=[self.kernel_size[3], self.kernel_size[3],
                                                       self.n_hidden[2],
                                                       self.n_hidden[3]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['enc_b3'] = tf.Variable(tf.zeros([self.n_hidden[3]], dtype=tf.float32))

        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.test_size, self.train_size], tf.float32), name='Coef')

        return all_weights

    # Building the encoder
    def encoder(self, X, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(X.get_shape().as_list())

        layer1 = tf.nn.bias_add(
            tf.nn.conv2d(X, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
            weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.nn.bias_add(
            tf.nn.conv2d(layer1, weights['enc_w1'], strides=[1, 1, 1, 1], padding='SAME'),
            weights['enc_b1'])
        layer2 = tf.nn.relu(layer2)
        layer3 = tf.nn.bias_add(
            tf.nn.conv2d(layer2, weights['enc_w2'], strides=[1, 2, 2, 1], padding='SAME'),
            weights['enc_b2'])
        layer3 = tf.nn.relu(layer3)
        latents = layer3
        print(layer3.shape)

        shapes.append(layer1.get_shape().as_list())
        shapes.append(layer2.get_shape().as_list())
        layer3_in = layer3

        latent = tf.nn.conv2d(layer3_in, weights['enc_w3'], strides=[1, 1, 1, 1], padding='SAME')
        latent = tf.nn.relu(latent)
        shapes.append(latent.get_shape().as_list())

        return latent, latents, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack(
            [tf.shape(self.x)[0], shape_de1[1], shape_de1[2], shape_de1[3]]), \
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)
        shape_de2 = shapes[1]
        layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights['dec_w1'], tf.stack(
            [tf.shape(self.x)[0], shape_de2[1], shape_de2[2], shape_de2[3]]), \
                                               strides=[1, 1, 1, 1], padding='SAME'), weights['dec_b1'])
        layer2 = tf.nn.relu(layer2)
        shape_de3 = shapes[0]
        layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights['dec_w2'], tf.stack(
            [tf.shape(self.x)[0], shape_de3[1], shape_de3[2], shape_de3[3]]), \
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b2'])
        layer3 = tf.nn.relu(layer3)
        recons = layer3,

        return recons

    def partial_fit(self):
        cost, Coef,_ = self.sess.run(
            (self.reconst_cost_x, self.Coef_test,self.train_step))
        self.iter = self.iter + 1
        return cost, Coef

    def pretrain_step(self):
        cost,_= self.sess.run(
            (self.reconst_cost_x,self.train_step_pretrain))
        self.iter = self.iter + 1
        return cost

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run(self.x_r, feed_dict={self.x: X})

    def transform(self, X, Y):
        return self.sess.run(self.AE, feed_dict={self.train: Y, self.test: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")


def thrC(C, ro=0.1):
    if ro < 1:
        N1 = C.shape[0]
        N2 = C.shape[1]
        Cp = np.zeros((N1, N2))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N2):
            cL1 = np.sum(S[:, i]).astype(np.float32)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def err_rate(gt_s, s):
    err_x = np.sum(gt_s[:] != s[:])
    missrate = err_x.astype(np.float32) / (gt_s.shape[0])
    return missrate


def testing(Img_test, Img_train, train_labels, test_labels, CAE, num_class, args):
    Img_test = np.array(Img_test)
    Img_test = Img_test.astype(np.float32)
    Img_train = np.array(Img_train)
    Img_train = Img_train.astype(np.float32)

    train_labels = np.array(train_labels[:])
    train_labels = train_labels - train_labels.min() + 1
    train_labels = np.squeeze(train_labels)

    test_labels = np.array(test_labels[:])
    test_labels = test_labels - test_labels.min() + 1
    test_labels = np.squeeze(test_labels)

    CAE.initlization()
    max_step = args.max_step  # 500 + num_class*25# 100+num_class*20
    pretrain_max_step = args.pretrain_step
    display_step = args.display_step  # max_step
    lr = 1.0e-4

    epoch = 0
    class_ = np.zeros(np.max(test_labels))
    prediction = np.zeros(len(test_labels))
    ACC = []
    Cost = []
    bestACC = 0
    perf_list = []
    fps_list = []
    pretrainepoch50time = 0
    fitepoch50time = 0
    while epoch < pretrain_max_step:
        epoch = epoch + 1
        pretrain_start_time = time.time()
        cost = CAE.pretrain_step()   #
        pretrain_end_time = time.time()
        pretrainepoch50time += (pretrain_end_time - pretrain_start_time)
        if epoch % display_step == 0:
            print("pretrtain epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))
            print("Pretrain 50 epoch cost ", pretrainepoch50time,
                  "s")  # Record the time spent on 50 epochs during pretrain
            pretrainepoch50time = 0

    while epoch < max_step:
        epoch = epoch + 1
        fit_start_time = time.time()
        cost, Coef = CAE.partial_fit()  #
        fit_end_time = time.time()
        fitepoch50time += (fit_end_time - fit_start_time)
        if epoch % display_step == 0:
            print("epoch: %.1d" % epoch, "cost: %.8f" % (cost / float(batch_size)))
            print("Fit 50 epoch cost ", fitepoch50time, "s")  # Record the time spent on 50 epochs during training
            perf = fitepoch50time / 50 * 1000
            perf_list.append(perf)
            perf_ = np.mean(perf_list)

            fps = float(batch_size) / perf
            fps_list.append(fps)
            fps_ = np.mean(fps_list)
            print("perf:  %.4f ms fps: %.4f" % (perf_, fps_))

            fitepoch50time = 0
            Coef = thrC(Coef)
            Coef = np.abs(Coef)
            for test_sample in range(0, len(test_labels)):
                x = Coef[test_sample, :]
                for l in range(1, np.max(test_labels) + 1):
                    l_idx = np.array([j for j in range(0, len(train_labels)) if train_labels[j] == l])
                    l_idx = l_idx.astype(int)
                    class_[int(l - 1)] = sum(np.abs(x[l_idx]))
                prediction[test_sample] = np.argmax(class_) + 1

            prediction = np.array(prediction)
            missrate_x = err_rate(test_labels, prediction)
            acc_x = 1 - missrate_x
            print("accuracy: %.4f" % acc_x)  # Get the prediction results and record the accuracy
            if acc_x > bestACC:
                bestACC = acc_x
                CAE.save_model()
            ACC.append(acc_x)
            Cost.append(cost / float(batch_size))
    if False:  # change to ture to save values in a mat file
        sio.savemat('./coef.mat', dict(ACC=ACC, Coef=Coef, Cost=Cost))

    return acc_x, Coef


def get_train_test_data(data, training_rate=0.8):
    '''
    Extracts features and labels from the dictionary "data," and splits the samples
    into training and testing sets.

    Input:
        data: dictionary containing two keys: {feature, Label}
            data['features'] : vectorized features (1024 x N)
            data['Label']   : groundtruth labels (1 x N)
        rate: ratio of the # of training samples to the total # of samples

    Output:
        training and testing sets.

    '''

    Label = data['Label']
    Label = np.squeeze(np.array(Label))
    training_size = int(training_rate * len(Label))

    perm = np.random.permutation(len(Label))  # Disorganize data
    training_idx = perm[:training_size]  # get train data
    testing_idx = perm[training_size:]  # get test data

    train_labels = Label[training_idx]  # get train label
    test_labels = Label[testing_idx]  # get test label

    I_test = []
    I_train = []
    img = data['features']
    training_img = img[:, training_idx]  # get train image
    testing_img = img[:, testing_idx]  # get test image

    for i in range(training_img.shape[1]):
        temp = cv2.resize(np.reshape(training_img[:, i], [16, 16]),
                          (32, 32))  # resize 16*16 to 32*32 as the standard input size
        I_train.append(temp)
    Img_train = np.transpose(np.array(I_train), [0, 2, 1])
    Img_train = np.expand_dims(Img_train[:], 3)

    for i in range(testing_img.shape[1]):
        temp = cv2.resize(np.reshape(testing_img[:, i], [16, 16]), (32, 32))
        I_test.append(temp)
    Img_test = np.transpose(np.array(I_test), [0, 2, 1])
    Img_test = np.expand_dims(Img_test[:], 3)

    return Img_train, Img_test, train_labels, test_labels, Label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='USPS', help='path of the dataset')
    parser.add_argument('--model', dest='model', default='USPS',
                        help='name of the model to be saved')
    parser.add_argument('--rate', dest='rate', type=float, default=0.8, help='Pecentage of samples ')
    parser.add_argument('--max_step', dest='max_step', type=int, default=3000, help='Max # training epochs')
    parser.add_argument('--pretrain_step', dest='pretrain_step', type=int, default=1000,
                        help='Max # of pretraining epochs ')
    parser.add_argument('--display_step', dest='display_step', type=int, default=50, help='frequency of reports')
    parser.add_argument('--data_url', dest='data_url', help='frequency of reports')
    parser.add_argument('--data_path', dest='data_path', default='./data/', help='train data')  # add

    args = parser.parse_args()

    # load face images and labels
    print(tf.__version__)
    datapath = args.data_path + args.mat + '.mat'  # add
    data = sio.loadmat(datapath)  # load data
    img = [[], [], [], [], [], [], [], [], [], []]
    label = [[], [], [], [], [], [], [], [], [], []]
    for i in range(data['Y'].shape[0]):  # Get the first two hundred samples for each category as dataset
        if len(label[data['Y'][i][0] - 1]) < 200:
            label[int(data['Y'][i][0] - 1)].append(data['Y'][i][0])
            img[int(data['Y'][i][0] - 1)].append(data['X'][i])
    label = np.array(label)
    img = np.array(img)
    label = label.reshape(1, -1)
    img = img.reshape(-1, 256).transpose(1, 0)
    data1 = {}
    data1['Label'] = label
    data1['features'] = img

    # Split the data into training and testing sets
    Im_train, Im_test, train_labels, test_labels, Label = get_train_test_data(data1,
                                                                              training_rate=args.rate)  # get train dataset and test dataset

    # face image clustering
    n_input = [32, 32]
    kernel_size = [5, 3, 3, 1]
    n_hidden = [10, 20, 30, 30]

    iter_loop = 0

    num_class = Label.max()
    batch_size = len(Label)
    training_size = len(train_labels)

    # These regularization values work best if the features are intensity values between 0-225
    reg1 = 1.0  # random.uniform(1, 10)
    reg2 = 8.0  # random.uniform(1, 10)

    model_path = './models/' + args.model + '.ckpt'

    logs_path = './logs'

    tf.reset_default_graph()
    CAE = ConvAE(train_data = Im_train.astype(np.float32),test_data=Im_test.astype(np.float32),n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                 kernel_size=kernel_size, batch_size=batch_size, train_size=training_size,model_path=model_path, restore_path=model_path,
                 logs_path=logs_path)  # build garph

    ACC, C = testing(Im_test, Im_train, train_labels, test_labels, CAE, num_class, args)  # training process