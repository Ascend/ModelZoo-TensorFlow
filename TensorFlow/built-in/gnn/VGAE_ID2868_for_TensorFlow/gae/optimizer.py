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
# Author: Tao Wu 

import tensorflow.compat.v1 as tf
from hyperparameters import FLAGS

# pylint:disable=missing-function-docstring
class OptimizerAE(object):
    """optimizer for GAE"""
    def __init__(self, model, labels, pos_weight, norm, use_dropout):
        with tf.name_scope('GAE'):
            preds = model(model.inputs, use_dropout=use_dropout)
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            logits=preds, labels=labels, pos_weight=pos_weight))
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = optimizer.minimize(self.cost)
        correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds), 0.5), tf.int32),
                                      tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


class OptimizerVAE(object):
    """optimizer for VGAE"""
    def __init__(self, model, labels, num_nodes, pos_weight, norm, use_dropout):
        with tf.name_scope('VGAE'):
            preds, z_mean, z_log_std = model(model.inputs, use_dropout=use_dropout)
        log_lik = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            logits=preds, labels=labels, pos_weight=pos_weight))
        kl = -(0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(
            1 + 2 * z_log_std - tf.square(z_mean) -tf.square(tf.exp(z_log_std)), 1))
        self.cost = tf.add(log_lik, kl, name='vae_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = optimizer.minimize(self.cost)
        correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds), 0.5), tf.int32),
                                      tf.cast(labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
