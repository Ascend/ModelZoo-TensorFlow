
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
#import moxing as mox
import copy
from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
import argparse
#npu
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# 禁用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

flags = tf.flags
flags.DEFINE_integer('nb_epochs', 20, 'train epoch=200')
FLAGS = flags.FLAGS

##################################################################################################
# 需要修改的参数
##################################################################################################
data_url = '/home/work/user-job-dir/code/load_data/'  #data所在的文件夹
datafile = 'ACM3025.mat'    #data的文件名
dataset = 'acm' # 数据集名称
trained = '/home/work/user-job-dir/code/trained' # checkpoint存放文件夹

#############################################################################################

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True #在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision") # 混合精度

# #profiling数据采集
# custom_op.parameter_map["profiling_mode"].b = True
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes("task_trace:training_trace")

# # 是否开启dump功能
# custom_op.parameter_map["enable_dump"].b = True
# # dump文件保存路径,系统参数默认配置情况下,生成的dump数据存放在/var/log/npu/ide_daemon/dump/{ dump_path }目录下
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/tmp")
# # dump哪些迭代的数据,不配置或者配置为None,表示dump所有迭代的数据。多个迭代用“|”分割,例如:0|5|10;也可以用"-"指定迭代范围,例如:0|3-5|10
# custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
# # dump模式,默认仅dump算子输出数据,还可以dump算子输入数据,取值:input/output/all
# custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

config.graph_options.rewrite_options.remapping = RewriterConfig.OFF #关闭remap开关

featype = 'fea'
checkpt_file = '{}/{}_allMP_multi_{}_.ckpt'.format(trained, dataset, featype)
print('model: {}'.format(checkpt_file))

batch_size = 1
nb_epochs = FLAGS.nb_epochs #20 #200
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

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_dblp(path):
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
print("++++++++++++++++++++++++++++++++loading data+++++++++++++++++++++++++++++++++++++++++++++")
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp(data_url + datafile)

print("++++++++++++++++++++++++++++++++load data successly+++++++++++++++++++++++++++++++++++++++")
if featype == 'adj':
    fea_list = adj_list

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

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
    print(bias_in_list)
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity,
                                                       training=is_train,
                                                       attn_drop=attn_drop, ffd_drop=ffd_drop
                                                       )

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

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        checkpt_file_final = ''
        for epoch in range(nb_epochs):
            checkpt_file = '{}/{}_allMP_multi_{}_{}.ckpt'.format(trained, dataset, featype, epoch)
            tr_step = 0

            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            dura = 0
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                start_time = time.time()
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                dura += (time.time() - start_time)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            # import pdb; pdb.set_trace()
            print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f , acc = %.5f | Val: loss = %.5f, acc = %.5f , TimeHistory = %.6f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step,dura))

            dura = 0
            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    print("++++++++++++++++++++++saving checkpoint+++++++++++++++++++++++++++++++++++++")
                    # print('loaddata:{},\ncheckpt_file:{}'.format(args.loaddatapath,checkpt_file))
                    print('loaddata:{},\ncheckpt_file:{}'.format('loaddata', checkpt_file))
                    checkpt_file_final = copy.deepcopy(checkpt_file)
                    saver.save(sess, checkpt_file)
                    print("++++++++++++++++++++++save checkpoint successfully+++++++++++++++++++++++++++++++++++++")
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        print("++++++++++++++++++++++restoring checkpoint+++++++++++++++++++++++++++++++++++++")
        saver.restore(sess, checkpt_file_final)
        print("++++++++++++++++++++++restore checkpoint successfully+++++++++++++++++++++++++++++++++++++")
        print('load model from : {}'.format(checkpt_file_final))
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
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
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
        from jhyexp import my_KNN, my_Kmeans#, my_TSNE, my_Linear

        my_KNN(xx, yy)
        my_Kmeans(xx, yy)

        sess.close()

#mox.file.copy_parallel('/home/work/user-job-dir/code/trained', 'obs://hannet/MA-HAN-tf-Mingze-06-03-11-36/output_bin')
