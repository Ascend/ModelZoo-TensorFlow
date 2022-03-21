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
import time
import numpy as np
import tensorflow as tf
import os
import moxing as mox
import copy
from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
import argparse
#npu
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# 禁用gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#####################################################################################################
#注意修改path
def load_data_dblp(path='/home/work/user-job-dir/code/load_data/ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()
if featype == 'adj':
    fea_list = adj_list

import scipy.sparse as sp

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    meta_path = '/home/work/user-job-dir/code/trained/acm_allMP_multi_fea_.ckpt.meta'
    ckpt = '/home/work/user-job-dir/code/trained/acm_allMP_multi_fea_.ckpt'
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, ckpt)
        # saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(ckpt))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],

                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            # logits_1 = sess.run([logits], feed_dict=fd)
            # print(logits_1)
            # logits_1.tofile("logits_1.bin")
            logits_1, loss_value_ts, acc_ts, jhy_final_embedding = sess.run([logits, loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        print('start knn, kmean.....')
        xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

        from numpy import linalg as LA

        # xx = xx / LA.norm(xx, axis=1)
        yy = y_test[test_mask]

        print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        from jhyexp import my_KNN, my_Kmeans  # , my_TSNE, my_Linear

        my_KNN(xx, yy)
        my_Kmeans(xx, yy)

        print(type(logits_1))
        logits_1.tofile("/home/work/user-job-dir/code/prediction/prediction.bin")

        sess.close()
mox.file.copy_parallel('/home/work/user-job-dir/code/prediction', 'obs://hannet/MA-HAN-tf-Mingze-06-03-11-36/output_bin')