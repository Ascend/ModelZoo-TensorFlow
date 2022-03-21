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
"""
main function for DL training.
"""
from __future__ import print_function
from npu_bridge.npu_init import *

import datetime
import os
import sys
import time
import math
import threading
from multiprocessing import Process

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as config
from data_utils import input_fn_tfrecord

#from models.WideDeep import WideDeep
from DCN_T import DCN_T

num_gpu = config.num_gpu
mode = 'train'
algo='DCN_T'


data_para = {
    'batch_size': int(sys.argv[1]),#config.batch_size,
    'num_gpu':num_gpu
}
train_para = {
    'pos_weight': 1.0,
    'n_epoch': int(sys.argv[4]),
    'train_per_epoch': config.train_size/5,
    'test_per_epoch': config.test_size,
    'batch_size': data_para['batch_size'],
    'early_stop_epochs': 500
}

record_path = str(sys.argv[5])
# set PIN model param
width = 1000
depth = 2
ls = [width] * depth
ls.append(1)
la = ['relu'] * depth
la.append(None)
lk = [.8] * depth
lk.append(1.)
model_param = {
    'init': 'xavier',
    'num_inputs': config.num_features,
    'input_dim': config.num_inputs,
    'real_inputs': config.num_features,
    'multi_hot_flags': config.multi_hot_flags,
    'multi_hot_len': config.multi_hot_len,
    'norm': False,
    'learning_rate': 5e-4,
    'embed_size': 80,
    'l2_v':1e-4,
    'layer_sizes': ls,
    'layer_acts': la,
    'layer_keeps': lk,
    'layer_l2': None,
    'net_sizes': [80],
    'net_acts': ['relu', None],
    'net_keeps': [0.8],
    'wide': True,
    'layer_norm': True,
    'sub_layer_norm': False
}


def write_log(log_path, _line, echo=False):
    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print(_line)


def metric(log_path, batch_auc, y, p, name='ctr', cal_prob=None):
    y = np.array(y)
    p = np.array(p)

    if cal_prob:
        if cal_prob <= 0 or cal_prob >1:
            raise ValueError('please ensure cal_prob is in (0,1]!')
        p /= (p + (1 -p) / cal_prob)
    auc = roc_auc_score(y, p)
    orilen = len(p)

    ind = np.where((p > 0) & (p < 1))[0]
    # print(len(ind))
    y = y[ind]
    p = p[ind]
    afterlen = len(p)
    # print('train auc: %g\tavg ctr: %g' % (batch_auc, y.mean()))

    ll = log_loss(y, p) * afterlen / orilen;
    q = y.mean()
    ne = ll / (-1 * q * np.log(q) - (1 - q) * np.log(1 -q))
    rig = 1 - ne

    if log_path:
        log = '%s\t%g\t%g\t%g\t%g' % (name, batch_auc, auc, ll, ne)
        write_log(log_path, log)
    print('avg %s on p: %g\teval auc: %g\tlog loss: %g\tne: %g\trig: %g' %
          (name, q, auc, ll, ne, rig))
    return auc


def evaluate_batch(sess, _model, num_gpu, batch_ids, batch_ws, batch_ys):
    if num_gpu == 1:
        model = _model
        feed_dict = {model.eval_id_hldr: batch_ids, model.eval_wt_hldr: batch_ws, model.eval_label: batch_ys}
        _preds_ = sess.run(fetches=model.eval_preds, feed_dict=feed_dict)
        batch_preds = [_preds_.flatten()]
    else:
        model = _model[0]
        feed_dict = {model.eval_id_hldr: batch_ids, model.eval_wt_hldr: batch_ws, model.eval_label: batch_ys}
        _preds_ = sess.run(fetches=model.eval_preds, feed_dict=feed_dict)
        batch_preds = [_preds_.flatten()]

    return batch_preds


def get_optimizer(optimizer_array, global_step):
    opt = optimizer_array[0].lower()
    #if algo == 'DCN_T':
    #   lr = tf.train.exponential_decay(learining_rate=optimizer_array[1], global_step=global_step, decay_rate=optimizer_array[3], decay_setps=optimizer_array[4], staircase=True)
    #else:
    lr = optimizer_array[1]
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        eps = optimizer_array[2]
        print("------------Adam: lr: {}; eps: {}".format(lr, eps))
        #
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        init_val = optimizer_array[2]
        return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=init_val)
    elif opt == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate=lr, initial_accumulator_value=optimizer_array[2],
                                      l1_regularization_strength=optimizer_array[3],
                                      l2_regularization_strength=optimizer_array[4])
#
def evaluate(sess, num_gpu,  model): # id_hldr, wt_hldr, eval_preds):
    preds = []
    labels = []
    line_cnt =0
    start_time = time.time()
    number_of_batches = ((train_para['test_per_epoch'] + train_para['batch_size'] - 1) /
                         train_para['batch_size'])
    print("%d batches in test set." % number_of_batches)
    #for _batch in range(1, number_of_batches + 1):
    epoch_finished = False
    test_dataset = input_fn_tfrecord(config.test_tag, record_path, config.batch_size / config.line_per_sample)
    test_iterator = test_dataset.make_initializable_iterator()
    next_element = test_iterator.get_next()
    sess.run([test_iterator.initializer])
    while not epoch_finished:
        # test_ids, test_wts, test_labels, epoch_finished = test_gen.next()
        try:
            test_batch_features = sess.run(next_element)
            test_ids = test_batch_features['feat_ids'].reshape((-1, config.num_inputs))
            test_wts = test_batch_features['feat_vals'].reshape((-1, config.num_inputs))
            test_labels = test_batch_features['label'].reshape((-1,))

            line_cnt += test_labels.shape[0]
            preds.append(np.squeeze(evaluate_batch(sess, model,  num_gpu, test_ids, test_wts, test_labels)))
            labels.append(np.squeeze(test_labels))
        except tf.errors.OutOfRangeError:
            print("end of test trainset")
            epoch_finished = True
            
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    print("evaluate time: %f sec" % (time.time() - start_time))
    return labels,  preds


def build_model(para_l2, _input_d):
    seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
    input_dim = config.num_features
    num_inputs = config.num_inputs

    model = DCN_T([input_dim, num_inputs, config.multi_hot_flags,
                   config.multi_hot_len],
                   _input_d,
                   [80, 3, [1024, 512, 256, 128], 'relu'],
                   ['uniform', -0.01, 0.01, seeds[4:14], None],
                   ['adam', 1e-4, 5e-8, 0.6, 5],
                   [0.7, 8e-5, 1e-8]
                   )
    #model = WideDeep([input_dim, num_inputs, config.multi_hot_flags,
    #              config.multi_hot_len],
    #              _input_d,
    #              [80, 3 [1024, 512, 256, 128], 'tanh'],
    #              ['uniform', -0.001, 0.001, seeds[4:14], None],
    #              ['adam', 1e-4, 5e-8, 0.6, 5],
    #              [0.7, 8e-5, 1e-8]
    #              )
    print('mode:%s, batch size: %d, buf size: %d, eval size: %d' %(
        mode, batch_size, buf_size, eval_size))

    write_log(log_file, model.log, True)
    return model

def average_gradients(gpu_grads):
    #if(len(gpu_grads) == 1):
    # return gpu_grads[0]

    avg_grads = []
    for grad_and_vars in zip(*gpu_grads):
        grads = []

        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        all_grad = tf.concat(grads, 0)
        avg_grad = tf.reduce_mean(all_grad, 0, keep_dims=False)

        v = grad_and_vars[0][1]
        grad_and_var = (avg_grad, v)
        avg_grads.append(grad_and_var)

    return avg_grads

def build_graph(para_l2, optimizer_array, input_data):
    # tf.reset_default_graph()
    with tf.device('/cpu:0'):
        with tf.variable_scope(tf.get_variable_scope()):
            global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                initializer=tf.constant_initializer(0), trainable=False)
            model = build_model(para_l2, input_data)
            opt = [get_optimizer(i, global_step) for i in optimizer_array]
            # tf.get_variable_scope().reuse_variables()
            # grads = opt.compute_gradients(model.loss)
            train_op = []
            if (len(opt) > 1):
                train_op.append(opt[0].minimize(loss=model.deep_loss, var_list=tf.get_collection('deep')))
                train_op.append(opt[1].minimize(loss=model.wide_loss, var_list=tf.get_collection('wide')))
            else:
                print('-------------------')
                loss_scale_manager = ExponentialUpdateLossScaleManager(
                     init_loss_scale= 2 ** 32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2,decr_ratio=0.5)
                opt[0] = NPULossScaleOptimizer(opt[0], loss_scale_manager)
                train_op.append(opt[0].minimize(loss=model.loss, global_step=global_step))
    return model, train_op


def build_graph_mgpu(num_gpu, optimizer_array, input_data):
    #tf.reset_default_graph()
    tower_grads = []
    models = []

    [input_lbl, input_id, input_wt] = input_data
    with tf.device('/cpu:0'):
        with tf.variable_scope(tf.get_variable_scope()):
            global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                initializer=tf.constant_initializer(0), trainable=False)
            opt = get_optimizer(optimizer_array, global_step)
            input_wt_s = tf.split(input_wt, num_gpu, 0)
            input_id_s = tf.split(input_id, num_gpu, 0)
            input_lbl_s = tf.split(input_lbl, num_gpu, 0)
            for i in xrange(num_gpu):
                with tf.device('/cpu:0'):
                    print('Deploying gpu:%d ...' % i)
                    with tf.name_scope('tower_%d' % i):
                        _input_d = [input_lbl_s[i], input_id_s[i], input_wt_s[i]]
                        model = build_model(_input_d)
                        models.append(model)
                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(model.loss)
                        tower_grads.append(grads)
    avgGrad_var_s = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(avgGrad_var_s, global_step=global_step)
    
    return models, apply_gradient_op

def train_batch(sess, num_gpu, _model, train_op):
    if num_gpu == 1:
        model = _model
        #fetche = [train_op]
        fetche = [i for i in train_op]
        if len(fetche) > 1:
            fetche += [model.deep_loss, model.wide_loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
            _, _, _deeploss_, _wideloss_, _log_loss_, _l2_loss_, _preds_, _train_labels=sess.run(fetches=fetche)
            _loss_ = _deeploss_
        else:
            fetche += [model.loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
            _, _loss_, _log_loss_, _l2_loss_, _preds_, _train_labels= sess.run(fetches=fetche)
        _train_preds = [_preds_.flatten()]
        # _train_labels = [_labels_.flatten()]
    else:
        fetches = []
        for i, model in enumerate(_model):
            fetches += [model.loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
        ret = sess.run(fetches=[train_op] + fetches)
        # print(ret)
        _loss_ = np.mean([ret[i] for i in range(1, len(ret), 5)])
        _log_loss_ = np.mean([ret[i] for i in range(2, len(ret), 5)])
        _l2_loss_ = np.mean([ret[i] for i in range(3, len(ret), 5)])
        _preds_ = [ret[i] for i in range(4, len(ret), 5)]
        _train_labels_ = [ret[i] for i in range(5, len(ret), 5)]
        _train_preds = [x.flatten() for x in _preds_]
        _train_labels = np.hstack(_train_labels_)
    return _loss_, _log_loss_, _l2_loss_, _train_preds, _train_labels

def create_dirs(dir):
    """create dir recursively as needed"""
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':

    # CUDA_VISIBLE_DEVICES=1 python -u dcn_main_record_multigpu_fb16.py 10000 dcn_fp16_learning_rate_1e-4 1e-4


    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    exp_tag = sys.argv[2]
    para_l2 = float(sys.argv[3])  # 1e-4
    input_dim = config.num_features
    num_inputs = config.num_inputs
    print("input_dim={}, num_inputs={}".format(input_dim, num_inputs))

    # timestamp = sys.argv[1]
    # tag = timestamp + '-' + algo

    tag = algo
    Base_path = config.BASE_DIR
    log_path = os.path.join(Base_path, 'log/')
    create_dirs(log_path)
    log_file = os.path.join(log_path, tag)
    pickle_model_path = os.path.join(Base_path,
                                     'model/pickle_model/')
    create_dirs(pickle_model_path)
    print("log file: ", log_file)

    batch_size = data_para['batch_size']
    buf_size = train_para['train_per_epoch']
    eval_size = data_para['batch_size']
    early_stop_epochs = train_para['early_stop_epochs']

    metric_best = 0
    metric_best_epoch = -1
    #optimizer_array = ['adam', 1e-4, 1e-8, 'mean']
    #optimizer_array = [['adam', 1e-4, 1e-8, 0.5, 5],["ftrl", 0.1, 1, 1e-8, 1e-8]]
    optimizer_array = [['adam', para_l2, 5e-8, 0.6, 5]]
    #



    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True

    global_start_time = time.time()
    with tf.device('/cpu:0'):
        #train_dataset = input_fn_tfrecord(config.train_record, batch_size=config.batch_size * num_gpu / config.line_per_sample,
        #                                  perform_shuffle=True, num_epochs=config.n_epoches)


        train_dataset = input_fn_tfrecord(config.train_tag, record_path, batch_size=int(sys.argv[1]) * num_gpu / config.line_per_sample,
                                          perform_shuffle=False, num_epochs=int(sys.argv[4]))
        # iterator = train_dataset.make_one_shot_iterator()
        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        input_data = [tf.reshape(next_element['label'], [-1]),
                      tf.reshape(next_element['feat_ids'], [-1, 39]),
                      tf.reshape(next_element['feat_vals'], [-1, 39])]

    if num_gpu == 1:
        model, opt = build_graph(para_l2, optimizer_array, input_data)
    elif num_gpu > 1:
        model, opt = build_graph_mgpu(num_gpu, optimizer_array, input_data)
    else:
        exit(0)
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    with tf.Session(config=npu_config_proto(config_proto=sess_config)) as sess:
        writer = tf.summary.FileWriter("lsp_model/", sess.graph)
        sess.run([iterator.initializer])
        sess.run(tf.global_variables_initializer())

        print('model initialized')

        if mode == 'train':
            
            start_time = time.time()
            est_epoch_batches = int((train_para['train_per_epoch'] +
                                     train_para['batch_size']*num_gpu - 1) / (train_para['batch_size']*num_gpu))
            est_tot_batches = train_para['n_epoch'] * est_epoch_batches
            _epoch = 1
            train_finished = False
            while _epoch < train_para['n_epoch'] +1 and not train_finished:
                _epoch_start_time = time.time()
                epoch_loss = []
                epoch_labels = []
                epoch_preds = []
                epoch_auc = -1
                epoch_finished = False
                epoch_sample_num = 0
                epoch_finished_batches = 0
                while not epoch_finished and not train_finished:
                    try:
                        _loss, _log_loss, _l2_loss, p, _labels = train_batch(sess, num_gpu, model, opt)
                        epoch_loss.append(_loss)
                        epoch_labels.append(_labels)
                        epoch_preds.extend(p)
                        epoch_finished_batches += 1
                        epoch_sample_num += _labels.shape[0]
                        if epoch_finished_batches % (100 / num_gpu) == 0:
                            if False:#_epoch == 10:
                                avg_loss = np.array(epoch_loss).mean()
                                epoch_auc = 0#roc_auc_score(epoch_labels, epoch_preds)
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                epoch_labels = []
                                epoch_preds = []
                                epoch_loss = []
                                print('epoch %3d/%3d - batch %5d: loss = %f, auc = %f | elapsed : %s, ETA : %s' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches, avg_loss, epoch_auc,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))
                                avg_loss = 0
                            else:
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                print('epoch %3d/%3d - batch %5d: | elapsed : %s, ETA : %s' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))
                        if epoch_finished_batches % est_epoch_batches == 0 or epoch_finished:
                            epoch_finished = True
                            print('epoch %d train time = %.3f sec, #train sample = %d' %
                                  (_epoch, time.time() - _epoch_start_time, epoch_sample_num))

                            # ======== comment this block if no testset available ========
                            print("== starting evaluate ==")
                            eval_labels, eval_preds = evaluate(sess, num_gpu, model)
                            eval_auc = metric(log_file, epoch_auc, eval_labels, eval_preds)
                            print("== finished evaluate ==")
                            # ============================================================
                            print('epoch %d total time = %s' %
                                  (_epoch, str(time.time() - _epoch_start_time)))

                            if eval_auc >= metric_best:
                                metric_best = eval_auc
                                metric_best_epoch = _epoch
                                '''
                                saver = tf.train.Saver()
                                saver.save(sess, Base_path + 'model/%s' % tag,
                                           global_step=_epoch,
                                           latest_filename='%s-checkpoint' % tag)
                                '''
                                print("current best auc: ", metric_best, " best_epoch: ", metric_best_epoch)
                            else:
                                if _epoch - metric_best_epoch >= early_stop_epochs:
                                    print("the model will be early stopped: current epoch:", _epoch)
                                    log_data = "best epoch: %d\t best performance:%g\n" % (metric_best_epoch, metric_best)
                                    log_data += "model_saved to %s\n" % (Base_path + 'model/%s' % tag)
                                    write_log(log_file, log_data, echo=True)

                                    print("save complete for epoch %d" % _epoch)
                                    train_finished = True
                                    break
                            _epoch += 1
                    except tf.errors.OutOfRangeError as e:
                        print("end of training dataset")
                        print("epoch %3d finished ..." % _epoch)
                        train_finished = True

            writer.close()

        elif mode == 'test':
            pass

        sess.close()