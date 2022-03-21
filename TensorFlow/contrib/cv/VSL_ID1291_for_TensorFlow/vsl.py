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

from npu_bridge.npu_init import *
from tensorflow.contrib import layers
import tensorflow as tf

class VarShapeLearner(object):
    def __init__(self, obj_res,
                 batch_size,
                 global_latent_dim,
                 local_latent_dim,
                 local_latent_num):
        # define model parameters
        self.obj_res = obj_res
        self.batch_size = batch_size
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.local_latent_num = local_latent_num

        # define input placeholder
        self.input_shape = [self.batch_size] + [self.obj_res]*3 + [1]
        self.x = tf.placeholder(tf.float32, self.input_shape)

        # create model and define its loss and optimizer
        self._model_create()
        self._model_loss_optimizer()

        # start tensorflow session
        self.saver = tf.train.Saver()

        config_proto = tf.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # 开启profiling采集
        custom_op.parameter_map["profiling_mode"].b = True
        # 仅采集任务轨迹数据
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/cache/profiling","task_trace":"on"}')
        # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/HwHiAiUser/output","task_trace":"on","training_trace":"on","fp_point":"resnet_model/conv2d/Conv2Dresnet_model/batch_normalization/FusedBatchNormV3_Reduce","bp_point":"gradients/AddN_70"}')
        config = npu_config_proto(config_proto=config_proto)
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

    # initialize model weights (using dictionary style)
    def _weights_init(self):
        # z_0, z_i parameters
        self.z_mean, self.z_logstd, self.z_all, self.kl_loss = ([0] * (self.local_latent_num + 1) for _ in range(4))

        # z_0 -> z_i parameters
        self.enc_zzi_fclayer1, self.enc_zzi_fclayer2 = [[0] * self.local_latent_num for _ in range(2)]

        # z_i -> z_{i+1} parameters
        self.enc_zizi_fclayer1, self.enc_zizi_fclayer2 = [[0] * (self.local_latent_num - 1) for _ in range(2)]

        # all -> z_i parameters
        self.enc_allzi_fclayer1, self.enc_allzi_fclayer2 = ([0] * self.local_latent_num for _ in range(2))

        # x -> z_0, z_i conv layers
        self.enc_conv1, self.enc_conv2, self.enc_conv3 = ([0] * (self.local_latent_num + 1) for _ in range(3))
        self.enc_fclayer1, self.enc_fclayer2 = ([0] * (self.local_latent_num + 1) for _ in range(2))

        # define all weights and biases of the network
        self.weights_all = dict()
        self.weights_all['W'] = {
            # input_shape x -> all_lat z_0, z_i (0, 1:local_lat_num)
            'enc_conv1': [tf.get_variable(name='enc_conv1', shape=[6, 6, 6, 1, 32],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_conv2': [tf.get_variable(name='enc_conv2', shape=[5, 5, 5, 32, 64],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_conv3': [tf.get_variable(name='enc_conv3', shape=[4, 4, 4, 64, 128],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_fc1'  : [tf.get_variable(name='enc_fc1', shape=[1024, 256],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_fc2'  : [tf.get_variable(name='enc_fc2', shape=[256, 100],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),

            # global_lat z_0 -> local_lat z_i
            'zzi_fc1' : [tf.get_variable(name='zzi_fc1', shape=[self.global_latent_dim, 100],
                                         initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zzi_fc2' : [tf.get_variable(name='zzi_fc2', shape=[100, 100],
                                         initializer=layers.xavier_initializer())]*self.local_latent_num,

            # local_lat z_i -> local_lat z_{i+1}
            'zizi_fc1': [tf.get_variable(name='zizi_fc1', shape=[self.local_latent_dim, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),
            'zizi_fc2': [tf.get_variable(name='zizi_fc2', shape=[100, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),

            # input_shape x -> global_lat z
            'z_mean'  : tf.get_variable(name='z_mean', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),
            'z_logstd': tf.get_variable(name='z_logstd', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),

            # merged [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
            'allzi_fc1':[tf.get_variable(name='allz1_fc1', shape=[200, 100],
                                         initializer=layers.xavier_initializer())]+
                        [tf.get_variable(name='allzi_fc1', shape=[300, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),
            'allzi_fc2': [tf.get_variable(name='allzi_fc2', shape=[100, 100],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zi_mean'  : [tf.get_variable(name='zi_mean', shape=[100, self.local_latent_dim],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zi_logstd': [tf.get_variable(name='zi_logstd', shape=[100, self.local_latent_dim],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,

            # combined lat [z_0, z_i] - > input_shape x
            'dec_fc1'  : tf.get_variable(name='dec_zfc1',
                                         shape=[self.global_latent_dim+self.local_latent_num*self.local_latent_dim,
                                                100 * (self.local_latent_num + 1)],
                                          initializer=layers.xavier_initializer()),
            'dec_fc2'  : tf.get_variable(name='dec_fc2', shape=[100 * (self.local_latent_num + 1), 1024],
                                         initializer=layers.xavier_initializer()),
            'dec_conv1': tf.get_variable(name='dec_conv1', shape=[4, 4, 4, 64, 128],
                                          initializer=layers.xavier_initializer()),
            'dec_conv2': tf.get_variable(name='dec_conv2', shape=[5, 5, 5, 32, 64],
                                          initializer=layers.xavier_initializer()),
            'dec_conv3': tf.get_variable(name='dec_conv3', shape=[6, 6, 6, 1, 32],
                                          initializer=layers.xavier_initializer())
        }

        self.weights_all['b'] = {
            # input_shape x -> all_lat z_0, z_i (0, 1:local_lat_num)
            'enc_conv1' : [tf.Variable(name='enc_conv1', initial_value=tf.zeros(32))]*(self.local_latent_num+1),
            'enc_conv2' : [tf.Variable(name='enc_conv2', initial_value=tf.zeros(64))]*(self.local_latent_num+1),
            'enc_conv3' : [tf.Variable(name='enc_conv3', initial_value=tf.zeros(128))]*(self.local_latent_num+1),
            'enc_fc1'   : [tf.Variable(name='enc_fc1', initial_value=tf.zeros(256))]*(self.local_latent_num+1),
            'enc_fc2'   : [tf.Variable(name='enc_fc2', initial_value=tf.zeros(100))]*(self.local_latent_num+1),

            # global_lat z_0 -> local_lat z_i
            'zzi_fc1': [tf.Variable(name='zzi_fc1', initial_value=tf.zeros(100))] * self.local_latent_num,
            'zzi_fc2': [tf.Variable(name='zzi_fc2', initial_value=tf.zeros(100))] * self.local_latent_num,

            # local_lat z_i -> local_lat z_{i+1}
            'zizi_fc1': [tf.Variable(name='zizi_fc1', initial_value=tf.zeros(100))] * (self.local_latent_num-1),
            'zizi_fc2': [tf.Variable(name='zizi_fc2', initial_value=tf.zeros(100))] * (self.local_latent_num-1),

            # local_lat z_i -> local_lat z_{i+1}
            'z_mean'  : tf.Variable(name='z_mean', initial_value=tf.zeros(self.global_latent_dim)),
            'z_logstd': tf.Variable(name='z_logstd', initial_value=tf.zeros(self.global_latent_dim)),

            # combined [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
            'allzi_fc1': [tf.Variable(name='allzi_fc1', initial_value=tf.zeros(100))]*self.local_latent_num,
            'allzi_fc2': [tf.Variable(name='allzi_fc2', initial_value=tf.zeros(100))] * self.local_latent_num,
            'zi_mean'  : [tf.Variable(name='zi_mean', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,
            'zi_logstd': [tf.Variable(name='zi_logstd', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,

            # combined lat [z_0, z_i] - > input_shape x
            'dec_fc1': tf.Variable(name='dec_fc1', initial_value=tf.zeros(100*(self.local_latent_num + 1))),
            'dec_fc2': tf.Variable(name='dec_fc2', initial_value=tf.zeros(1024)),
            'dec_conv1': tf.Variable(name='dec_conv1', initial_value=tf.zeros(64)),
            'dec_conv2': tf.Variable(name='dec_conv2', initial_value=tf.zeros(32)),
            'dec_conv3': tf.Variable(name='dec_conv3', initial_value=tf.zeros(1)),
        }

    # use re-parametrization trick
    def _sampling(self, z_mean, z_logstd, latent_dim):
        epsilon = tf.random_normal((self.batch_size, latent_dim))
        return z_mean + tf.exp(z_logstd) * epsilon

    # define inference model q(z_0:n|x)
    def _inf_model(self, weights, biases):
        # input_shape x -> local_lat z_i
        for i in range(self.local_latent_num + 1):
            self.enc_conv1[i] = tf.nn.relu(tf.nn.conv3d(self.x, weights['enc_conv1'][i],
                                                        strides=[1, 2, 2, 2, 1], padding='VALID')
                                                        + biases['enc_conv1'][i])
            self.enc_conv2[i] = tf.nn.relu(tf.nn.conv3d(self.enc_conv1[i], weights['enc_conv2'][i],
                                                        strides=[1, 2, 2, 2, 1], padding='VALID')
                                                        + biases['enc_conv2'][i])
            self.enc_conv3[i] = tf.nn.relu(tf.nn.conv3d(self.enc_conv2[i], weights['enc_conv3'][i],
                                                        strides=[1, 1, 1, 1, 1], padding='VALID')
                                                        + biases['enc_conv3'][i])
            self.enc_conv3[i] = tf.reshape(self.enc_conv3[i], [self.batch_size, 1024])
            self.enc_fclayer1[i] = tf.nn.relu(tf.matmul(self.enc_conv3[i], weights['enc_fc1'][i])
                                              + biases['enc_fc1'][i])
            self.enc_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_fclayer1[i], weights['enc_fc2'][i])
                                              + biases['enc_fc2'][i])
        # sample global latent variable
        self.z_mean[0] = tf.matmul(self.enc_fclayer2[0], weights['z_mean']) + biases['z_mean']
        self.z_logstd[0] = tf.matmul(self.enc_fclayer2[0], weights['z_logstd']) + biases['z_logstd']
        self.z_all[0] = self._sampling(self.z_mean[0], self.z_logstd[0], self.global_latent_dim)

        for i in range(self.local_latent_num):
            # z_0 -> z_i
            self.enc_zzi_fclayer1[i] = tf.nn.relu(tf.matmul(self.z_all[0], weights['zzi_fc1'][i])
                                                  + biases['zzi_fc1'][i])
            self.enc_zzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_zzi_fclayer1[i], weights['zzi_fc2'][i])
                                                  + biases['zzi_fc2'][i])

            if i == 0:  # sampling z_1
                self.enc_allzi_fclayer1[i] = tf.nn.relu(tf.matmul(tf.concat([self.enc_zzi_fclayer2[i], self.enc_fclayer2[i+1]], axis=1),
                                                                  weights['allzi_fc1'][i]) + biases['allzi_fc1'][i])
                self.enc_allzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_zzi_fclayer1[i],
                                                                  weights['allzi_fc2'][i]) + biases['allzi_fc2'][i])

                self.z_mean[1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_mean'][i]) + biases['zi_mean'][i]
                self.z_logstd[1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_logstd'][i]) + biases['zi_logstd'][i]
                self.z_all[1] = self._sampling(self.z_mean[1], self.z_logstd[1],self.local_latent_dim)
            else:   # sampling z_i (i >= 1)
                self.enc_zizi_fclayer1[i-1] = tf.nn.relu(tf.matmul(self.z_all[i], weights['zizi_fc1'][i-1])
                                                      + biases['zizi_fc1'][i-1])
                self.enc_zizi_fclayer2[i-1] = tf.nn.relu(tf.matmul(self.enc_zizi_fclayer1[i-1], weights['zizi_fc2'][i-1])
                                                      + biases['zizi_fc2'][i-1])
                self.enc_allzi_fclayer1[i] = tf.nn.relu(tf.matmul(tf.concat([self.enc_zzi_fclayer2[i], self.enc_fclayer2[i+1], self.enc_zizi_fclayer2[i-1]], axis=1),
                                                                  weights['allzi_fc1'][i]) + biases['allzi_fc1'][i])
                self.enc_allzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_allzi_fclayer1[i], weights['allzi_fc2'][i])
                                                        + biases['allzi_fc2'][i])
                self.z_mean[i+1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_mean'][i]) + biases['zi_mean'][i]
                self.z_logstd[i+1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_logstd'][i]) + biases['zi_logstd'][i]
                self.z_all[i+1] = self._sampling(self.z_mean[i+1], self.z_logstd[i+1], self.local_latent_dim)

        # concatenate all latent codes
        self.latent_feature = tf.concat([self.z_mean[i] for i in range(self.local_latent_num + 1)], axis=1)


    # define generative model p(x|z_0:n)
    def _gen_model(self, weights, biases):
        dec_fclayer1 = tf.nn.relu(tf.matmul(self.latent_feature , weights['dec_fc1']) + biases['dec_fc1'])
        dec_fclayer2 = tf.nn.relu(tf.matmul(dec_fclayer1 , weights['dec_fc2']) + biases['dec_fc2'])
        dec_fclayer2 = tf.reshape(dec_fclayer2, [self.batch_size, 2, 2, 2, 128])
        dec_conv1    = tf.nn.relu(tf.nn.conv3d_transpose(dec_fclayer2, weights['dec_conv1'],
                                  output_shape=[self.batch_size, 5, 5, 5, 64],
                                  strides=[1, 1, 1, 1, 1],padding='VALID') + biases['dec_conv1'])
        dec_conv2    = tf.nn.relu(tf.nn.conv3d_transpose(dec_conv1, weights['dec_conv2'],
                                  output_shape=[self.batch_size, 13, 13, 13, 32],
                                  strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv2'])
        dec_conv3    = tf.nn.sigmoid(tf.nn.conv3d_transpose(dec_conv2, weights['dec_conv3'],
                                     output_shape=[self.batch_size, 30, 30, 30, 1],
                                     strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv3'], name='output')
        return dec_conv3

    # create model
    def _model_create(self):
        # load defined network structure
        self._weights_init()
        network_weights = self.weights_all

        # learn latent parameters from inference network
        self._inf_model(network_weights['W'], network_weights['b'])

        # reconstruct training data from learned latent features
        self.x_rec = self._gen_model(network_weights['W'], network_weights['b'])

    # define VSL loss and optimizer
    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_mean(self.x * tf.log(1e-5 + self.x_rec)
                                       +(1-self.x) * tf.log(1e-5 + 1 - self.x_rec), axis=(1, 2))

        # define kl loss
        for i in range(self.local_latent_num + 1):
            self.kl_loss[i] = -0.5 * tf.reduce_sum(1 + 2 * self.z_logstd[i] - tf.square(self.z_mean[i]) - tf.square(tf.exp(self.z_logstd[i])), axis=1)

        self.kl_loss_all = tf.add_n(self.kl_loss) / (self.local_latent_num + 1)

        # total loss = kl loss + rec loss
        self.loss = tf.reduce_mean(self.rec_loss + 0.001*self.kl_loss_all)


        # gradient clipping to avoid nan
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5)
        gradients = optimizer.compute_gradients(self.loss)
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
        self.optimizer = optimizer.apply_gradients(clipped_gradients)

    # train model on mini-batch
    def model_fit(self, x):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x})
        return cost
