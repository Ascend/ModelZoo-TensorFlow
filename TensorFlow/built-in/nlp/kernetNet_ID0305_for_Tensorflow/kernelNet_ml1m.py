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


import numpy as np
import tensorflow as tf
from time import time
import sys
from dataLoader import loadData
import os
import argparse
from npu_bridge.npu_init import *

from tfdeterminism import patch
import random

seed = 1627543871
patch()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', dest='save_dir', default='./test/out/checkpoints')
parser.add_argument('--data_path', dest='data_path', default='./ml-1m', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_fp32_to_fp16', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')
parser.add_argument('--npu_loss_scale', dest='npu_loss_scale', type=int, default=1)
parser.add_argument('--mode', dest='mode', default='train', choices=('train', 'test', 'train_and_eval'))
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10)
args = parser.parse_args()

# load data
database = args.data_path
save_path = args.save_dir

tr, vr = loadData(os.path.join(database, 'ratings.dat'), delimiter='::',
                  seed=seed, transpose=True, valfrac=0.1)

tm = np.greater(tr, 1e-12).astype('float32')  # masks indicating non-zero entries
vm = np.greater(vr, 1e-12).astype('float32')

n_m = tr.shape[0]  # number of movies
n_u = tr.shape[1]  # number of users (may be switched depending on 'transpose' in loadData)

# Set hyper-parameters
n_hid = 500
# lambda_2 = float(sys.argv[1]) if len(sys.argv) > 1 else 70.
# lambda_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.013
lambda_2 = 70.
lambda_s = 0.013
n_layers = 2
output_every = 50  # evaluate performance on test set; breaks l-bfgs loop

n_epoch = n_layers * 10 * output_every

verbose_bfgs = True
use_gpu = True
if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Input placeholders
R = tf.placeholder("float", [None, n_u])


# define network functions
def kernel(u, v):
    """
    Sparsifying kernel function

    :param u: input vectors [n_in, 1, n_dim]
    :param v: output vectors [1, n_hid, n_dim]
    :return: input to output connection matrix
    """
    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist ** 2)
    return hat


def kernel_layer(x, n_hid=500, n_dim=5, activation=tf.nn.sigmoid, lambda_s=lambda_s,
                 lambda_2=lambda_2, name=''):
    """
    a kernel sparsified layer

    :param x: input [batch, channels]
    :param n_hid: number of hidden units
    :param n_dim: number of dimensions to embed for kernelization
    :param activation: output activation
    :param name: layer name for scoping
    :return: layer output, regularization term
    """

    # define variables
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-4))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-4))
        b = tf.get_variable('b', [n_hid])

    # compute sparsifying kernel
    # as u and v move further from each other for some given pair of neurons, their connection
    # decreases in strength and eventually goes to zero.
    w_hat = kernel(u, v)

    # compute regularization terms
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    # compute output
    W_eff = W * w_hat
    y = tf.matmul(x, W_eff) + b
    y = activation(y)
    return y, sparse_reg_term + l2_reg_term


# Instantiate network
y = R
reg_losses = None
for i in range(n_layers):
    y, reg_loss = kernel_layer(y, n_hid, name=str(i))
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
prediction, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
reg_losses = reg_losses + reg_loss

# Compute loss (symbolic)
diff = tm * (R - prediction)
sqE = tf.nn.l2_loss(diff)
loss = sqE + reg_losses

# Instantiate L-BFGS Optimizer
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': output_every,
                                                                  'disp': verbose_bfgs,
                                                                  'maxcor': 10},
                                                   method='L-BFGS-B')

# Training and validation loop
saver = tf.train.Saver()
init = tf.global_variables_initializer()

# Start a new TensorFlow session.
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")

best_validation_rmse = 1.0
epoch_best_validation_rmse = 0
training_time_sum = 0
epoch_count = 0

with tf.Session(config=config) as sess:
    sess.run(init)
    tf.io.write_graph(sess.graph_def, 'checkpoints', 'train.pbtxt')
    for i in range(int(n_epoch / output_every)):
        # Start timing
        strat_time = time()
        optimizer.minimize(sess, feed_dict={R: tr})  # do maxiter optimization steps
        pre = sess.run(prediction, feed_dict={R: tr})  # predict ratings
        avg_time_per_step = (time() - strat_time)
        print("### avg_time_per_epoch: ", avg_time_per_step, " seconds/epoch ### ")
        training_time_sum += avg_time_per_step
        epoch_count += 1

        error = (vm * (np.clip(pre, 1., 5.) - vr) ** 2).sum() / vm.sum()  # compute validation error
        error_train = (tm * (np.clip(pre, 1., 5.) - tr) ** 2).sum() / tm.sum()  # compute train error

        print('.-^-._' * 12)
        print('epoch:', i, 'validation rmse:', np.sqrt(error), 'train rmse:', np.sqrt(error_train))
        print('.-^-._' * 12)

        if np.sqrt(error) < best_validation_rmse:
            best_validation_rmse = np.sqrt(error)
            saver.save(sess=sess, save_path=save_path)
            epoch_best_validation_rmse = i
        print("The best validation rmse: ", best_validation_rmse, " from epoch ", epoch_best_validation_rmse)

    print("########################################### The Result Start ##########################################")
    print("[Accuracy INFO]  The best validation rmse: ", best_validation_rmse, " from epoch ",
          epoch_best_validation_rmse, " ###")
    print("[Performance INFO]  The all  ", epoch_count, "  epoch use training_time_sum: ", training_time_sum,
          " seconds ### ")
    print("[Performance INFO]  The avg_time_per_epoch: ", training_time_sum / epoch_count, " seconds ### ")
    print("[Performance INFO]  The avg_epoch_per_secs: ", epoch_count / training_time_sum, " epochs ### ")
    print("########################################### The Result End   ##########################################")
    # with open('summary_ml1m.txt', 'a') as file:
    #     for a in sys.argv[1:]:
    #         file.write(a + ' ')
    #     file.write(str(np.sqrt(error)) + ' ' + str(np.sqrt(error_train))
    #                + ' ' + str(seed) + '\n')
    #     file.close()
