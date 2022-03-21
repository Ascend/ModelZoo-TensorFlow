#
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
#
# Deep Multimodal Subspace Clustering Networks
# https://arxiv.org/abs/1804.06498
# Mahdi Abavisani
# mahdi.abavisani@rutgers.edu
# Built upon https://github.com/panji1990/Deep-subspace-clustering-networks
import time

from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse


def next_batch(data_, _index_in_epoch, batch_size, num_modalities, _epochs_completed):
    _num_examples = data_['0'].shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        for i in range(0, num_modalities):
            data_[str(i)] = data_[str(i)][perm]

        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    data = {}
    for i in range(0, num_modalities):
        data[str(i)] = data_[str(i)][start:end]
    return data, _index_in_epoch, _epochs_completed


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, num_modalities=2, batch_size=256, reg=None, denoise=False, \
                 model_path=None, restore_path=None, logs_path='./logs'):
        # n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg,
        self.model_path = model_path
        self.restore_path = restore_path
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.iter = 0
        self.num_modalities = num_modalities
        self.learning_rate = tf.placeholder(tf.float32, [],
                                            name='learningRate')
        self.weights = self._initialize_weights()
        self.x = {}

        # model

        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])

        if not denoise:
            x_input = self.x
            latents, shape = self.encoder(x_input, self.weights, self.num_modalities)

        z = {}
        z_c = {}
        latent_c = {}
        Coef = self.weights['Coef']

        for i in range(0, self.num_modalities):
            modality = str(i)
            z[modality] = tf.reshape(latents[modality], [batch_size, -1])
            z_c[modality] = tf.matmul(Coef, z[modality])
            latent_c[modality] = tf.reshape(z_c[modality], tf.shape(latents[modality]))

        self.Coef = Coef
        self.z = z
        self.x_r = self.decoder(latent_c, self.weights, self.num_modalities, shape)

        self.saver = tf.train.Saver()

        self.reconst_cost_x = 0.6 * tf.losses.mean_squared_error(self.x_r['0'], self.x['0'])
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.reconst_cost_x = self.reconst_cost_x + 0.1 * tf.losses.mean_squared_error(self.x_r[modality], \
                                                                                           self.x[modality])
        self.cost = self.reconst_cost_x

        tf.summary.scalar("l2_loss", self.cost)
        
        self.merged_summary_op = tf.summary.merge_all()

        self.global_steps = tf.Variable(0, trainable=False)
        self.loss = self.cost
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**14, incr_every_n_steps=1000,
                                                               decr_every_n_nan_or_inf=1, decr_ratio=0.5)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        self.optimizer = opt.minimize(self.loss, global_step=self.global_steps)

        self.ls = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
        self.overflow_status_reduce_all = tf.get_default_graph().get_tensor_by_name("overflow_status_reduce_all:0")
        tf.set_random_seed(0)

        self.init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()

        for i in range(0, self.num_modalities):
            modality = str(i)
            with tf.variable_scope(modality):
                all_weights[modality + '_enc_w0'] = tf.get_variable(modality + "_enc_w0",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                           self.n_hidden[0]],
                                                                    initializer=tf.keras.initializers.glorot_normal())


                all_weights[modality + '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32), \
                                                                name=modality + '_enc_b0')

                all_weights[modality + '_enc_w1'] = tf.get_variable(modality + "_enc_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32), \
                                                                name=modality + '_enc_b1')

                all_weights[modality + '_enc_w2'] = tf.get_variable(modality + "_enc_w2",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[2]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32), \
                                                                name=modality + '_enc_b2')

                all_weights[modality + '_dec_w0'] = tf.get_variable(modality + "_dec1_w0",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[3]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32), \
                                                                name=modality + '_dec_b0')

                all_weights[modality + '_dec_w1'] = tf.get_variable(modality + "_dec1_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32), \
                                                                name=modality + '_dec_b1')

                all_weights[modality + '_dec_w2'] = tf.get_variable(modality + "_dec1_w2",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                           self.n_hidden[0]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32), \
                                                                name=modality + '_dec_b2')

        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], dtype=tf.float32), name='Coef')

        return all_weights

    # Building the encoder
    def encoder(self, X, weights, num_modalities):
        shapes = []
        latents = {}
        # Encoder Hidden layer with relu activation #1
        shapes.append(X['0'].get_shape().as_list())
        for i in range(0, num_modalities):
            modality = str(i)
            layer1 = tf.nn.bias_add(tf.nn.conv2d(X[modality], weights[modality + '_enc_w0'], \
                                                 strides=[1, 2, 2, 1], padding='SAME'), weights[modality + '_enc_b0'])
            layer1 = tf.nn.relu(layer1)
            layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights[modality + '_enc_w1'], \
                                                 strides=[1, 1, 1, 1], padding='SAME'), weights[modality + '_enc_b1'])
            layer2 = tf.nn.relu(layer2)
            layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights[modality + '_enc_w2'], \
                                                 strides=[1, 2, 2, 1], padding='SAME'), weights[modality + '_enc_b2'])
            layer3 = tf.nn.relu(layer3)
            latents[modality] = layer3
            if i == 0:
                shapes.append(layer1.get_shape().as_list())
                shapes.append(layer2.get_shape().as_list())
                shapes.append(layer3.get_shape().as_list())

        return latents, shapes

    # Building the decoder
    def decoder(self, z, weights, num_modalities, shapes):
        recons = {}
        # Encoder Hidden layer with relu activation #1
        for i in range(0, num_modalities):
            modality = str(i)
            shape_de1 = shapes[2]
            layer1 = tf.add(tf.nn.conv2d_transpose(z[modality], weights[modality+'_dec_w0'], \
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de1[1], shape_de1[2], \
                                                             shape_de1[3]]), strides=[1, 2, 2, 1], padding='SAME'), \
                            weights[modality + '_dec_b0'])
            layer1 = tf.nn.relu(layer1)
            shape_de2 = shapes[1]
            layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights[modality+'_dec_w1'], \
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de2[1], shape_de2[2], \
                                                             shape_de2[3]]), strides=[1, 1, 1, 1], padding='SAME'), \
                            weights[modality + '_dec_b1'])
            layer2 = tf.nn.relu(layer2)
            shape_de3 = shapes[0]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights[modality+'_dec_w2'], \
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de3[1], shape_de3[2], \
                                                             shape_de3[3]]), strides=[1, 2, 2, 1], padding='SAME'), \
                            weights[modality + '_dec_b2'])
            layer3 = tf.nn.relu(layer3)
            recons[modality] = layer3

        return recons

    def partial_fit(self, X, lr):
        feed_dict = {}
        feed_dict[self.learning_rate] = lr
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _, ls, ov = self.sess.run((self.cost, self.merged_summary_op, self.optimizer, self.ls,
                                                 self.overflow_status_reduce_all), feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, ls, ov

    def initlization(self):
        self.sess.run(self.init)

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")


def train_face(Img, CAE, n_input, batch_size, num_modalities):
    it = 0
    display_step = 50
    _index_in_epoch = 0
    _epochs = 0
    init_time = time.time()
    stat_time = time.time()
    CAE.restore()
    min_loss = 0.0015
    lr = 1e-3
    # train the network
    while True:
        batch, _index_in_epoch, _epochs = next_batch(Img, _index_in_epoch, batch_size, num_modalities, _epochs)
        for i in range(0, num_modalities):
            start_time = time.time()
            batch[str(i)] = np.reshape(batch[str(i)], [batch_size, n_input[0], n_input[1], 1])
        cost, ls, ov = CAE.partial_fit(batch, lr)
        if not ov:
            print("overflow ls is %d" % ls)
        it = it + 1
        avg_cost = cost/batch_size
        cost_time = time.time() - start_time
        if it % display_step == 0:
            print("epoch: %.1d" % _epochs)
            print("cost: %.8f" % avg_cost)
            # print("average time cost: %.4f" % ((time.time() - stat_time)/display_step))
            print("perf: %.4f" % cost_time)
            stat_time = time.time()
            if avg_cost < min_loss:
                min_loss = avg_cost
                CAE.save_model()
            if (avg_cost < 0.001) and (time.time()-init_time > 2000):
                CAE.save_model()
                print("finish")
                break


if __name__ == '__main__':
    tf.reset_default_graph()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='YaleB', help='path of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='# of epoch')
    parser.add_argument('--model', dest='model', default='mymodel',
                        help='name of the model to be saved')

    args = parser.parse_args()
    datapath = './Data/' + args.mat + '.mat'
    data = sio.loadmat(datapath)
    batch_size = 2424
    num_modalities1 = data['num_modalities'][0][0]
    Img = {}
    X = {}
    np.random.seed(0)

    for i in range(0, num_modalities1):
        I = []
        modality = str(i)
        img = data['modality_'+modality][:batch_size, :]
        for i in range(img.shape[1]):
            temp = np.reshape(img[:, i], [32, 32])
            I.append(temp)
        Img[modality] = np.transpose(np.array(I), [0, 2, 1])  # TODO: might need adding expand_dims

    Label = data['Label'][:batch_size]
    Label = np.array(Label)

    n_input = [32, 32]
    kernel_size = [5, 3, 3, 3]
    n_hidden = [10, 20, 30, 30]

    lr = 1.0e-3  # learning rate
    model_path = './models/' + args.model + '.ckpt'
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, kernel_size=kernel_size,
                 batch_size=batch_size, model_path=model_path, restore_path=model_path, num_modalities=num_modalities1)

    train_face(Img, CAE, n_input, batch_size, num_modalities1)
