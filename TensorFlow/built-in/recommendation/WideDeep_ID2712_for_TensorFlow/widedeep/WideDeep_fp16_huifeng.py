# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DeepFM related model"""
from __future__ import print_function
import os
import sys
import pickle
import tensorflow as tf
from widedeep.tf_util import activate
from widedeep import features
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.hccl import hccl_ops
from npu_bridge.npu_init import *

rank_size = os.getenv('RANK_SIZE')
rank_size = 1 if not rank_size else int(rank_size)

class WideDeep:
    """support performing mean pooling Operation on multi-hot feature.
    dataset_argv: e.g. [10000, 17, [False, ...,True,True], 10]
    """
    def __init__(self, graph, architect_argv, ptmzr_argv,
                 reg_argv, _input_data, _eval_data=None, loss_mode='full',
                 cross_layer=False, batch_norm=False):
        self.graph = graph
        with self.graph.as_default():
            wide_inputs, deep_inputs = features.get_feature_inputs(_input_data[0])
            self.wide_layer = tf.transpose(tf.concat([tf.transpose(x) for x in wide_inputs], axis=0))
            self.deep_layer = tf.transpose(tf.concat([tf.transpose(x) for x in deep_inputs], axis=0))
            self.labels = tf.squeeze(tf.cast(_input_data[1], tf.float32), axis=1)
            layer_dims, act_func = architect_argv
            keep_prob, _lambda = reg_argv

            self.embed_dim = self.deep_layer.get_shape()[1]
            self.all_layer_dims = [self.embed_dim] + layer_dims + [1]
            self.log = ('embedding layer: %d\nlayers: %s\nactivate: %s\n'
                        'keep_prob: %g\nl2(lambda): %g\n' %
                        (self.embed_dim, self.all_layer_dims, act_func, keep_prob, _lambda))
            with tf.variable_scope("wide_embeddings", reuse=tf.AUTO_REUSE):
                self.wide_b = tf.get_variable('wide_b', [1],
                                            initializer = tf.random_uniform_initializer(-0.01, 0.01),
                                            dtype=tf.float32,
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES, "wide", "wide_bias"])


            with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
                self.h_w, self.h_b = [], []
                for i in range(len(self.all_layer_dims) - 1):
                    self.h_w.append(tf.get_variable('h%d_w' % (i + 1), shape=self.all_layer_dims[i: i + 2],
                                                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                                    dtype=tf.float32,
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_wts"]))
                    self.h_b.append(
                        tf.get_variable('h%d_b' % (i + 1), shape=[self.all_layer_dims[i + 1]], initializer=tf.zeros_initializer,
                                        dtype=tf.float32,
                                        collections=[tf.GraphKeys.GLOBAL_VARIABLES, "deep", "mlp_bias"]))

            print("_input_data[1]== self.lables", self.labels)

            wideout = self.wide_forward(self.wide_layer)
            y = self.forward(
                self.deep_layer, act_func, keep_prob, training=True,
                cross_layer=cross_layer, batch_norm=batch_norm)
            y = y + wideout
            self.train_preds = tf.sigmoid(y, name='predicitons')

            basic_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=self.labels)

            self.wide_loss = tf.reduce_mean(basic_loss)
            self.deep_loss = tf.reduce_mean(basic_loss) + _lambda * tf.nn.l2_loss(self.deep_layer)

            self.l2_loss = tf.constant([0]) #self.loss
            self.log_loss = basic_loss

            if _eval_data:
                eval_wide_inputs, eval_deep_inputs = features.get_feature_inputs(_eval_data[0])
                self.eval_wide_layer = tf.concat(eval_wide_inputs, axis=1)
                self.eval_deep_layer = tf.concat(eval_deep_inputs, axis=1)
                self.eval_labels = tf.squeeze(tf.cast(_eval_data[1], tf.float32), axis=1)
 
                eval_wideout= self.wide_forward(self.eval_wide_layer)
                eval_y = self.forward(
                    self.eval_deep_layer, act_func, keep_prob, training=False,
                    cross_layer=cross_layer, batch_norm=batch_norm)
                eval_y = eval_y + eval_wideout
 
                self.eval_preds = tf.sigmoid(eval_y, name='eval_predictions')

            opt_deep, lr_deep, eps_deep, decay_rate_deep, decay_step_deep = ptmzr_argv[0]
            opt_wide, wide_lr, wide_dc, wide_l1, wide_l2 = ptmzr_argv[1]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #loss_scale_manager = FixedLossScaleManager(loss_scale=1000.0)
                #loss_scale_manager2 = FixedLossScaleManager(loss_scale=1000.0)
                loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
                loss_scale_manager2 = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
                self.wide_ptmzr = tf.train.FtrlOptimizer(learning_rate=wide_lr, initial_accumulator_value=wide_dc,
                                                    l1_regularization_strength=wide_l1,
                                                    l2_regularization_strength=wide_l2)
                if rank_size > 1:
                    self.wide_ptmzr = tf.train.FtrlOptimizer(learning_rate=wide_lr * rank_size, initial_accumulator_value=wide_dc,
                                                    l1_regularization_strength=wide_l1,
                                                    l2_regularization_strength=wide_l2)
                    ############## for hccl ###################
                    #self.wide_ptmzr = npu_distributed_optimizer_wrapper(self.wide_ptmzr)
                    ############## for hccl ###################
                    #self.wide_ptmzr = NPULossScaleOptimizer(self.wide_ptmzr, loss_scale_manager, is_distributed=True)
                    grads_w = self.wide_ptmzr.compute_gradients(self.wide_loss, var_list=tf.get_collection('wide'))
                else:
                    self.wide_ptmzr = NPULossScaleOptimizer(self.wide_ptmzr, loss_scale_manager)
                    self.wide_ptmzr = self.wide_ptmzr.minimize(self.wide_loss,var_list=tf.get_collection("wide"))
                self.deep_optimzer = tf.train.AdamOptimizer(learning_rate=lr_deep, epsilon=eps_deep)
                if rank_size > 1:
                    self.deep_optimzer = tf.train.AdamOptimizer(learning_rate=lr_deep * rank_size, epsilon=eps_deep)
                    ############## for hccl ###################
                    #self.deep_optimzer = npu_distributed_optimizer_wrapper(self.deep_optimzer)
                    ############## for hccl ###################
                    #self.deep_optimzer = NPULossScaleOptimizer(self.deep_optimzer, loss_scale_manager2, is_distributed=True)
                    grads_d = self.deep_optimzer.compute_gradients(self.deep_loss, var_list=tf.get_collection('deep'))
                else:
                    self.deep_optimzer = NPULossScaleOptimizer(self.deep_optimzer, loss_scale_manager2)
                    self.deep_optimzer = self.deep_optimzer.minimize(self.deep_loss,var_list=tf.get_collection("deep"))
                if rank_size > 1:
                    ############## for hccl ###################
               #     self.train_op = tf.group(self.deep_optimzer, self.wide_ptmzr)
                    ############## for hccl ###################
                    avg_grads_w = []
                    avg_grads_d = []
                    for grad, var in grads_w:
                        avg_grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                        avg_grads_w.append((avg_grad, var))
                    for grad, var in grads_d:
                        avg_grad = hccl_ops.allreduce(grad, "sum") if grad is not None else None
                        avg_grads_d.append((avg_grad, var))
                    apply_gradient_op_w = self.wide_ptmzr.apply_gradients(avg_grads_w)
                    apply_gradient_op_d = self.deep_optimzer.apply_gradients(avg_grads_d)
                    self.train_op = tf.group(apply_gradient_op_d, apply_gradient_op_w)
                else:
                    self.train_op = tf.group(self.deep_optimzer, self.wide_ptmzr)


    def wide_forward(self, wide_part):
        wide_output = tf.reshape((tf.reduce_sum(wide_part, axis=1) + self.wide_b), shape=[-1, ],
                                 name="wide_out")
        return wide_output

    def forward(self, deep_part, act_func, keep_prob,
                training, cross_layer=False, batch_norm=False):
        hidden_output = tf.reshape(deep_part, [-1, self.embed_dim])
        cross_layer_output = None
        for i in range(len(self.h_w)):
            if training:
                hidden_output = tf.matmul(npu_ops.dropout(activate(act_func, hidden_output), keep_prob=keep_prob), self.h_w[i])
            else:
                hidden_output = tf.matmul(activate(act_func, hidden_output), self.h_w[i])
            hidden_output = hidden_output + self.h_b[i]

            if batch_norm:
                hidden_output = tf.layers.batch_normalization(
                    hidden_output, training=training)
            if cross_layer_output is not None:
                cross_layer_output = tf.concat(
                    [cross_layer_output, hidden_output], 1)
            else:
                cross_layer_output = hidden_output

            if cross_layer and i == len(self.h_w) - 2:
                hidden_output = cross_layer_output
        return tf.reshape(hidden_output, [-1, ])

    def dump(self, model_path):
        var_map = {'W': self.fm_w.eval(),
                   'V': self.fm_v.eval(),
                   'b': self.fm_b.eval()}

        for i, (h_w_i, h_b_i) in enumerate(zip(self.h_w, self.h_b)):
            var_map['h%d_w' % (i+1)] = h_w_i.eval()
            var_map['h%d_b' % (i+1)] = h_b_i.eval()

        pickle.dump(var_map, open(model_path, 'wb'))
        print('model dumped at %s' % model_path)


