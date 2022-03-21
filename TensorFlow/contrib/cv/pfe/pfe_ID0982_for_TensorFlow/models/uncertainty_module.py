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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

from npu_bridge.npu_init import *


batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': True,
    'updates_collections': None,
    # 'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    'variables_collections': [ tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES ],
}

batch_norm_params_sigma = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': False,
    'updates_collections': None,
    # 'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],}
    'variables_collections': [ tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES ],}


def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):

    num_channels = x.shape[-1].value
    # with tf.variable_scope('scale_and_shift'):
        # gamma = tf.get_variable('alpha', (),
    with tf.compat.v1.variable_scope('scale_and_shift'):
        gamma = tf.compat.v1.get_variable('alpha', (),
                        initializer=tf.constant_initializer(gamma_init),
                        regularizer=slim.l2_regularizer(0.0),
                        dtype=tf.float32)
        # beta = tf.get_variable('gamma', (),
        beta = tf.compat.v1.get_variable('gamma', (),
                        initializer=tf.constant_initializer(beta_init),
                        dtype=tf.float32)
        x = gamma * x +  beta

        return x   
    

def inference(inputs, embedding_size, phase_train, 
        weight_decay=5e-4, reuse=None, scope='UncertaintyModule'):
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu):
        # with tf.variable_scope(scope, [inputs], reuse=reuse):
        with tf.compat.v1.variable_scope(scope, [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('UncertaintyModule input shape:', [dim.value for dim in inputs.shape])

                net = slim.flatten(inputs)

                net = slim.fully_connected(net, embedding_size, scope='fc1',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, 
                    activation_fn=tf.nn.relu)


                log_sigma_sq = slim.fully_connected(net, embedding_size, scope='fc_log_sigma_sq',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_sigma,
                    activation_fn=None)
          
                # Share the gamma and beta for all dimensions
                log_sigma_sq = scale_and_shift(log_sigma_sq, 1e-4, -7.0)

                # Add epsilon for sigma_sq for numerical stableness                
                # log_sigma_sq = tf.log(1e-6 + tf.exp(log_sigma_sq))
                log_sigma_sq = tf.compat.v1.log(1e-6 + tf.exp(log_sigma_sq))

    return log_sigma_sq
