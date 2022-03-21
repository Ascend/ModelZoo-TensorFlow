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
from __future__ import print_function
from npu_bridge.npu_init import *

import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import tensorflow as tf
from tf_util import build_optimizer, init_var_map, \
    get_field_index, get_field_num, split_mask, split_param, sum_multi_hot, \
    activate


class DCN_T:

    def __init__(self, dataset_argv, input_data, architect_argv, init_argv, ptmzr_argv, reg_argv, loss_mode='full',
                 merge_multi_hot=False, batch_norm=False):
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        (features_size, fields_num, self.multi_hot_flags, self.multi_hot_len) = dataset_argv

        self.one_hot_flags = [not flag for flag in self.multi_hot_flags]

        embedding_size, num_cross_layer, deep_layers, act_func = architect_argv

        keep_prob, _lambda, l1_lambda = reg_argv

        self.num_onehot = sum(self.one_hot_flags)
        self.num_multihot = sum(self.multi_hot_flags) / self.multi_hot_len

        if merge_multi_hot:
            self.embedding_dim = (self.num_multihot +self.num_onehot) * embedding_size
        else:
            self.embedding_dim = fields_num * embedding_size

        # currently no output layer which is different from DNN
        self.all_deep_layer = [self.embedding_dim] +deep_layers

        self.log = ('input dim: %d\n'
                    'num inputs: %d\n'
                    'embed size(each): %d\n'
                    'embedding layer: %d\n'
                    'num cross layer: %d\n'
                    'deep layers: %s\n'
                    'activate: %s\n'
                    'keep_prob: %g\n'
                    'l2(lambda): %g\n'
                    'merge_multi_hot: %s\n' %
                    (features_size, fields_num, embedding_size, self.embedding_dim, num_cross_layer,
                    self.all_deep_layer,
                    act_func, keep_prob, _lambda, merge_multi_hot))
        print("-----------------  args: \n{}".format( self.log ))


        init_acts = [('cross_w', [num_cross_layer, self.embedding_dim], 'random'),
                     ('cross_b', [num_cross_layer, self.embedding_dim], 'random'),
                     ('embed', [features_size, embedding_size], 'random')]
        
        # add deep layer to init
        for i in range(len(self.all_deep_layer) - 1):
            init_acts.extend([('h%d_w' %(i + 1), self.all_deep_layer[i: i + 2], 'random'),
                              ('h%d_b' %(i + 1), [self.all_deep_layer[i + 1]], 'random')])
        
        # print(init_acts)

        var_map, log = init_var_map(init_argv, init_acts)

        self.log += log

        #
        #
        #
        self.embed_v = tf.get_variable('V', shape=[features_size, embedding_size],
                                    initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    dtype=tf.float32)
        self.cross_w = tf.get_variable('W', shape=[num_cross_layer, self.embedding_dim],
                                    initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]),
                                    dtype=tf.float32)
        self.cross_b = tf.get_variable('b', [num_cross_layer, self.embedding_dim], initializer=tf.zeros_initializer, dtype=tf.float32)

        self.h_w = []
        self.h_b = []
        for i in range(len(self.all_deep_layer) -1):
            #
            #
            self.h_w.append(tf.get_variable('h%d_w' % (i+1), shape=self.all_deep_layer[i: i+2], initializer = tf.random_uniform_initializer(init_argv[1], init_argv[2]), dtype=tf.float32))
            self.h_b.append(tf.get_variable('h%d_b' % (i+1), shape=[self.all_deep_layer[i+1]], initializer = tf.zeros_initializer, dtype = tf.float32))

        self.wt_hldr = input_data[2] #
        self.id_hldr = input_data[1] #
        self.lbl_hldr = input_data[0] #

        vx_embed = self.construct_embedding(self.wt_hldr, self.id_hldr, merge_multi_hot)

        #

        xl, final_hl = self.forward(vx_embed, num_cross_layer, act_func, keep_prob, training=True,
                                    batch_norm=batch_norm)

        #
        #
        # for debug
        x_stack = final_hl

        #
        #

        init_acts_final = [('out_w', [int(x_stack.shape[1]), 1], 'random'),
                           ('out_b', [1], 'zero')]

        var_map, log = init_var_map(init_argv, init_acts_final)

        self.log += log

        self.out_w = tf.get_variable('out_w',
                                     shape=[int(x_stack.shape[1]), 1],
                                     initializer=tf.random_uniform_initializer(init_argv[1], init_argv[2]), dtype=tf.float32)
        self.out_b = tf.get_variable('out_b', shape=[1], initializer=tf.zeros_initializer, dtype=tf.float32)

        #
        #

        y = self.final_forward(x_stack, self.out_w, self.out_b, act_func)

        self.train_preds = tf.sigmoid(y, name='predictions')

        #
        #
        #
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=self.lbl_hldr), name='loss')
        #
        self.loss = loss + tf.contrib.layers.l1_regularizer(l1_lambda)(self.cross_w) + \
                           tf.contrib.layers.l1_regularizer(l1_lambda)(self.cross_b) + \
                           _lambda * tf.nn.l2_loss(self.embed_v)
        #
        print("-----------  l1_lambda: {}; _lambda: {};".format( l1_lambda, _lambda ))

        self.log_loss = loss
        self.l2_loss = tf.constant([0])
        self.eval_wt_hldr = tf.placeholder(tf.float32, [None, fields_num], name='wt')
        self.eval_id_hldr = tf.placeholder(tf.int32, [None, fields_num], name='id')
        self.eval_label = tf.placeholder(tf.float32, [None, ], name='label')
        eval_vx_embed = self.construct_embedding(self.eval_wt_hldr, self.eval_id_hldr, merge_multi_hot)
        eval_xl, eval_final_hl = self.forward(eval_vx_embed, num_cross_layer, act_func, keep_prob,
                                              training=False,
                                              batch_norm=batch_norm)
        #eval_x_stack = tf.concat([eval_xl, eval_final_hl], 1)
        eval_x_stack = eval_final_hl
        eval_y = self.final_forward(eval_x_stack, self.out_w, self.out_b, act_func)
        self.eval_preds = tf.sigmoid(eval_y, name="predictionNode")

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        #   self.optmzr, log = build_optimizer(ptmzr_argv, self.loss)
        #self.log += log

        #update_ops = tf.get_collection(tf.GraphKeys,UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        #   learining_rate = tf.train.exponential_decay(learning_rate=ptmzr_argv[1], global_step=self.global_step, decay_rate=ptmzr_argv[3], decay_steps=ptmzr_argv[4], staircase=True)
        #   self.optmzr = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=ptmzr_argv[2]).minimize(self.loss)
        #   log = 'optimizer: %s,learining_rate: %g, epsilon: %g' % (ptmzr_argv[0]锛宲tmzr_argv[1]锛宲tmzr_argv[2])
        #self.log += log

    # construct the embedding layer
    def construct_embedding(self, wt_hldr, id_hldr, merge_multi_hot=False):
        mask = tf.expand_dims(wt_hldr, 2)
        if merge_multi_hot and self.num_multihot > 0:
            # *_hot_mask is weight(values that follow the ids in the datasetm defferent from weight of param) that used
            one_hot_mask, multi_hot_mask = split_mask(
                mask, self.multi_hot_flags,self.num_multihot)

            one_hot_v, multi_hot_v = split_param(
                self.embed_v, id_hldr, self.multi_hot_flags)

            # fm part (reduce multi-hot vector's length to k*1)
            multi_hot_vx = sum_multi_hot(
                multi_hot_v, multi_hot_mask, self.num_multihot)
            one_hot_vx = tf.multiply(one_hot_v, one_hot_mask)
            vx_embed = tf.concat([one_hot_vx, multi_hot_vx], axis=1)
        else:
            # [batch, input_dim4lookup, embed_size]
            # vx_embed = tf.multiply(tf.gather(self.embed_v, id_hldr), mask)
            vx_embed = tf.multiply(tf.nn.embedding_lookup(self.embed_v, id_hldr), mask)
        return vx_embed
    #

    def forward(self, vx_embed, num_cross_layer, act_func, keep_prob, training=True, batch_norm=False):
        # embedding layer
        x0 = tf.reshape(vx_embed, [-1, self.embedding_dim])

        # print('x0 shape: %s' % x0.shape)
        # cross layer
        xl = x0
        for i in range(num_cross_layer):
            #xlw = tf.tensordot(xl, self.cross_w[i], axes=1)
            #print('xlw shape: %s' % xlw.shape)

            xlw = tf.reduce_sum(tf.multiply(xl, self.cross_w[i]), 1)
            #print('xlw shape: %s' % xlw.shape)
            xl = x0 * tf.expand_dims(xlw, -1) + self.cross_b[i] + xl
            xl.set_shape((None, self.embedding_dim))

        # print('xl shape: %s' % xl.shape)

        # get final hidden layer output
        final_hl = self.deep_forward(vx_embed, self.h_w, self.h_b, act_func, keep_prob, training, batch_norm)

        # print('hidden layer shape: %s' % final_hl)

        return xl, final_hl

    def deep_forward(self, vx_embed, h_w, h_b, act_func, keep_prob, training, batch_norm=False):

        hidden_output = tf.reshape(vx_embed, [-1, self.embedding_dim])

        for i in range(len(h_w)):
            if training:
                # TODO: batch norm should be after relu
                # if batch_norm:

                hidden_output = tf.matmul(hidden_output, h_w[i]) + h_b[i]
                if batch_norm:
                    print("setting bn for training stage")
                    hidden_output = tf.layers.batch_normalization(
                        hidden_output, training=True, reuse=False, name="bn_%d" % i
                    )
                hidden_output = activate(act_func, hidden_output)
                hidden_output = npu_ops.dropout(hidden_output, keep_prob=keep_prob)
            else:
                hidden_output = tf.matmul(hidden_output, h_w[i]) + h_b[i]
                if batch_norm:
                    print("setting bn for testing stage")
                    hidden_output = tf.layers.batch_normalization(
                        hidden_output, training=False, reuse=True, name="bn_%d" % i
                    )
                hidden_output = activate(act_func, hidden_output)

        return hidden_output

    def final_forward(self,final_layer, out_w, out_b, act_func):
        hidden_output = tf.matmul(activate(act_func, final_layer), out_w) +out_b

        return tf.reshape(hidden_output, [-1])

        