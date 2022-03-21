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
import scipy.io as sio
import argparse
from metrics import *


class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1=1.0, re_constant2=1.0, batch_size=100, reg=None, \
                 denoise=False, restore_path=None, logs_path='./logs', num_modalities=2):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.restore_path = restore_path
        self.iter = 0
        self.num_modalities = num_modalities
        weights = self._initialize_weights()
        self.x = {}
        
        # input required to be fed
        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])

        self.learning_rate = tf.placeholder(tf.float32, [],
                                        name='learningRate')

        if not denoise:
            x_input = self.x
            latents, shape = self.encoder(x_input, weights, self.num_modalities)

        Coef = weights['Coef']
        Coef = Coef - tf.diag(tf.diag_part(Coef))
        z = {}
        z_c = {}
        latent_c = {}

        for i in range(0, self.num_modalities):
            modality = str(i)
            z[modality] = tf.reshape(latents[modality], [batch_size, -1])
            z_c[modality] = tf.matmul(Coef, z[modality])
            latent_c[modality] = tf.reshape(z_c[modality], tf.shape(latents[modality]))

        self.Coef = Coef
        self.z = z
        self.z_c = z_c

        self.x_r = self.decoder(latent_c, weights, self.num_modalities, shape)

        self.reconst_cost_x = 0.6 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r['0'], self.x['0']), 2.0))
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.reconst_cost_x = self.reconst_cost_x +  0.1*tf.reduce_sum(tf.pow(tf.subtract(self.x_r[modality], \
                                                                                              self.x[modality]), 2.0))

        tf.summary.scalar("recons_loss", self.reconst_cost_x)

        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef, 2.0))

        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses)

        self.selfexpress_losses = 0.3*tf.reduce_sum(tf.pow(tf.subtract(self.z['0'], self.z_c['0']), 2.0))
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.selfexpress_losses = self.selfexpress_losses + 0.05*tf.reduce_sum(tf.pow(tf.subtract(self.z[modality],
                                                                                               self.z_c[modality]), 2.0))
        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost_x + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        
        self.merged_summary_op = tf.summary.merge_all()

        loss_scale_manager = FixedLossScaleManager(loss_scale=2 ** 7, enable_overflow_check=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        self.optimizer = opt.minimize(self.loss)
        tf.set_random_seed(0)

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
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

                all_weights[modality + '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32),
                                                                name=modality + '_enc_b0')

                all_weights[modality + '_enc_w1'] = tf.get_variable(modality + "_enc_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32),
                                                                name=modality + '_enc_b1')

                all_weights[modality + '_enc_w2'] = tf.get_variable(modality + "_enc_w2",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[2]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32),
                                                                name=modality + '_enc_b2')

                all_weights[modality + '_dec_w0'] = tf.get_variable(modality + "_dec1_w0",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[3]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32),
                                                                name=modality + '_dec_b0')

                all_weights[modality + '_dec_w1'] = tf.get_variable(modality + "_dec1_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32),
                                                                name=modality + '_dec_b1')

                all_weights[modality + '_dec_w2'] = tf.get_variable(modality + "_dec1_w2",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                           self.n_hidden[0]],
                                                                    initializer=tf.keras.initializers.glorot_normal())
                all_weights[modality + '_dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                                name=modality + '_dec_b2')

        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], dtype=tf.float32),
                                          name='Coef')

        return all_weights

    # Building the encoder
    def encoder(self, X, weights, num_modalities):
        shapes = []
        latents = {}
        # Encoder Hidden layer with relu activation #1
        shapes.append(X['0'].get_shape().as_list())
        for i in range(0, num_modalities):
            modality = str(i)
            layer1 = tf.nn.bias_add(tf.nn.conv2d(X[modality], weights[modality + '_enc_w0'], strides=[1, 2, 2, 1], \
                                                 padding='SAME'), weights[modality + '_enc_b0'])
            layer1 = tf.nn.relu(layer1)
            layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights[modality + '_enc_w1'], strides=[1, 1, 1, 1], \
                                                 padding='SAME'), weights[modality + '_enc_b1'])
            layer2 = tf.nn.relu(layer2)
            layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights[modality + '_enc_w2'], strides=[1, 2, 2, 1], \
                                                 padding='SAME'), weights[modality + '_enc_b2'])
            layer3 = tf.nn.relu(layer3)
            latents[modality] = layer3
            print(layer3.shape)
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
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de1[1], \
                                                             shape_de1[2], shape_de1[3]]), strides=[1, 2, 2, 1], \
                                                   padding='SAME'), weights[modality+'_dec_b0'])
            layer1 = tf.nn.relu(layer1)
            shape_de2 = shapes[1]
            layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights[modality+'_dec_w1'], \
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de2[1], shape_de2[2], \
                                                             shape_de2[3]]), strides=[1, 1, 1, 1], padding='SAME'), \
                            weights[modality+'_dec_b1'])
            layer2 = tf.nn.relu(layer2)
            shape_de3 = shapes[0]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights[modality+'_dec_w2'], \
                                                   tf.stack([tf.shape(self.x['0'])[0], shape_de3[1], shape_de3[2], \
                                                             shape_de3[3]]), strides=[1, 2, 2, 1], padding='SAME'), \
                            weights[modality+'_dec_b2'])
            layer3 = tf.nn.relu(layer3)
            recons[modality] = layer3

        return recons

    def partial_fit(self, X, lr):
        feed_dict = {}
        feed_dict[self.learning_rate] = lr
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _, Coef = self.sess.run(
                (self.loss, self.merged_summary_op, self.optimizer, self.Coef), feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost, Coef
    
    def initlization(self):
        self.sess.run(self.init)

    def restore(self):
        print(self.restore_path)
        self.saver.restore(self.sess, self.restore_path)
        print("model restored")


def test_face(Img, Label, CAE, num_class, num_modalities):
    
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)

    for j in range(0, num_modalities):
        modality = str(j)
        Img[modality] = np.array(Img[modality])
        Img[modality] = Img[modality].astype(float)

    label = np.array(Label[:])
    label = label - label.min() + 1
    label = np.squeeze(label)

    CAE.initlization()
    CAE.restore()

    max_step = 800
    display_step = 10
    lr = 1.0e-3
    epoch = 0
    max_acc = 0
    record_start = True
    record_end = True
    while epoch < max_step:
        epoch = epoch + 1
        cost, Coef = CAE.partial_fit(Img, lr)

        if record_start:
            start_time = time.time()
            record_start = False

        if epoch % display_step == 0 and (epoch>=200):
            if record_end:
                print("time used per step: %.4f" % ((time.time()-start_time)/199))
                record_end = False

            print("epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(batch_size)))
            Coef = thrC(Coef, alpha)
            y_x, _ = post_proC(Coef, label.max())
            missrate_x, nmi, ari = err_rate(label, y_x)
            acc = 1 - missrate_x
            if acc > max_acc:
                max_acc = acc
            print("accuracy: %.2f%%" % (acc*100))
    print("Max ACC: %.2f%%" % (max_acc*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='YaleB', help='path of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=150000, help='# of epoch')
    parser.add_argument('--model', dest='model', default='multimodal',
                        help='name of the model to be saved')

    args = parser.parse_args()
    # load face images and labels
    datapath = './Data/' + args.mat + '.mat'
    data = sio.loadmat(datapath)

    num_modalities1 = data['num_modalities'][0][0]
    print(num_modalities1)
    Img = {}
    X = {}
    I = []
    for i in range(0, num_modalities1):
        I = []
        modality = str(i)
        img = data['modality_'+modality]
        for i in range(img.shape[1]):
            temp = np.reshape(img[:, i], [32, 32])
            I.append(temp)
        Img [modality] = np.transpose(np.array(I), [0, 2, 1])
        Img[modality] = np.expand_dims(Img[modality][:], 3)
        print(Img[modality].shape)

    Label = data['Label']
    Label = np.array(Label)
    np.random.seed(0)

    # face image clustering
    n_input = [32, 32]
    kernel_size = [5, 3, 3, 3]
    n_hidden = [10, 20, 30, 30]
    
    all_subjects = Label.max()

    avg = []
    med = []
    
    iter_loop = 0
    num_class = all_subjects
    batch_size = Img['0'].shape[0]
    print(batch_size)
    reg1 = 1.0
    reg2 = 1.0 * 10 ** (num_class / 10.0 - 3.0)

    restore_path = './models/' + args.model + '.ckpt'
    logs_path = './logs'
    tf.reset_default_graph()
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                 kernel_size=kernel_size, batch_size=batch_size, restore_path=restore_path, logs_path=logs_path, \
                 num_modalities=num_modalities1)

    test_face(Img, Label, CAE, num_class, num_modalities1)
