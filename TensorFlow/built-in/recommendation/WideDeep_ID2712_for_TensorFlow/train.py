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
"""
main function for DL training.
"""
from __future__ import print_function

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
import configs.config as config
from widedeep.data_utils import input_fn_tfrecord
from npu_bridge.estimator.npu import util
from widedeep.WideDeep_fp16_huifeng import WideDeep
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce

###add by daihongtao
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
#from hccl.manage.api import get_rank_size
#from hccl.manage.api import get_rank_id
###end
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
from hccl.split.api import set_split_strategy_by_idx
set_split_strategy_by_idx([2,64,67,88])

rank_size = os.getenv('RANK_SIZE')
rank_size = 1 if not rank_size else int(rank_size)
##single p时，需要采设置修改
rank_id = os.getenv('DEVICE_INDEX')
rank_id = 0 if not rank_id else int(rank_id)
#rank_id = 0 if rank_size == 1 else get_rank_id()
#num_gpu = config.num_gpu
num_gpu = rank_size
mode = 'train'
algo='WideDeep'


###dump
from npu_bridge.estimator.npu.npu_config import DumpConfig



data_para = {
    'batch_size': int(config.batch_size),#config.batch_size,
    'eval_batch_size': int(config.eval_batch_size),#config.batch_size,
    'num_gpu': num_gpu
}
train_para = {
    'pos_weight': 1.0,
    'n_epoch': config.n_epoches,
    'train_per_epoch': config.train_size,
    'test_per_epoch': config.test_size,
    'batch_size': data_para['batch_size'],
    'eval_batch_size': data_para['eval_batch_size'],
    'early_stop_epochs': 50,
    # 'iterations_per_loop': config.iterations_per_loop
}
#if rank_size > 1:
#    train_para = {
#    'pos_weight': 1.0,
#    'n_epoch': config.n_epoches_8p,
#    'train_per_epoch': config.train_size,
#    'test_per_epoch': config.test_size,
#    'batch_size': data_para['batch_size'],
#    'eval_batch_size': data_para['eval_batch_size'],
#    'early_stop_epochs': 50,
    # 'iterations_per_loop': config.iterations_per_loop
#    }

def write_log(log_path, _line, echo=False):
    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print(_line)


def metric(log_path, batch_auc, y, p, name='ctr', cal_prob=None):
    y = np.array(y)
    p = np.array(p)

    if cal_prob:
        if cal_prob <= 0 or cal_prob > 1:
            raise ValueError('please ensure cal_prob is in (0,1]!')
        p /= (p + (1 - p) / cal_prob)
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
    ne = ll / (-1 * q * np.log(q) - (1 - q) * np.log(1 - q))
    rig = 1 - ne

    if log_path:
        log = '%s\t%g\t%g\t%g\t%g' % (name, batch_auc, auc, ll, ne)
        write_log(log_path, log)
    print('avg %s on p: %g\teval auc: %g\tlog loss: %g\tne: %g\trig: %g' %
          (name, q, auc, ll, ne, rig))
    return auc

####global_step
def get_optimizer(optimizer_array, global_step):
    opt = optimizer_array[0].lower()
    #if algo == 'DCN_T':
    lr = tf.train.exponential_decay(learning_rate=optimizer_array[1], global_step=global_step, decay_rate=optimizer_array[3], decay_steps=optimizer_array[4], staircase=True)
    #else:
    #lr = optimizer_array[1]
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        eps = optimizer_array[2]
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        init_val = optimizer_array[2]
        return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=init_val)
    elif opt == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate=lr, initial_accumulator_value=optimizer_array[2],l1_regularization_strength=optimizer_array[3],l2_regularization_strength=optimizer_array[4])

def evaluate(sess, model): # id_hldr, wt_hldr, eval_preds):
    preds = []
    labels = []
    start_time = time.time()
    number_of_batches = ((train_para['test_per_epoch'] + train_para['eval_batch_size'] - 1) /
                         train_para['eval_batch_size'])
    print("%d batches in test set." % number_of_batches)
    #for _batch in range(1, number_of_batches + 1):
    iter = 0
    print("evaluate wile start time: %f sec" % (time.time() - start_time))
    while iter < number_of_batches:
        iter = iter + 1
        try:
            _preds, _labels = sess.run(fetches=[model.eval_preds, model.eval_labels])
            batch_preds = [_preds.flatten()]
            preds.append(np.squeeze(batch_preds))
            labels.append(_labels)
        except tf.errors.OutOfRangeError:
            print("evaluate end of test trainset time: %f sec" % (time.time() - start_time))
            print("end of test trainset")
            break
    print("evaluate labels time: %f sec" % (time.time() - start_time))
    labels = np.hstack(labels)
    print("evaluate preds time: %f sec" % (time.time() - start_time))
    preds = np.hstack(preds)
    print("evaluate time: %f sec" % (time.time() - start_time))
    return labels,  preds


def build_model(graph, _input_d, _eval_d):
    seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]

    #drop1 0.80768 drop0.8 50epoch 0.808996

    model = WideDeep(graph,
                     [[1024, 1024, 1024, 1024, 1024], 'relu'],
                     [['adam', 1e-4, 9e-8, 0.8, 5], ['ftrl', 0.06, 1, 0, 0]],
                     [1.0, 9e-6],
                     _input_d, _eval_d
                     )

    print('mode:%s, batch size: %d, buf size: %d, eval size: %d' % (
        mode, batch_size, buf_size, eval_size))

    write_log(log_file, model.log, True)
    return model

def average_gradients(gpu_grads):
    #if(len(gpu_grads) == 1):
    #    return gpu_grads[0]

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

def build_graph(graph, input_data, eval_data,rank_id):
    # tf.reset_default_graph()
    with tf.device(0):
    ####### for hccl ########
  #  with tf.device(rank_id):
    ####### for hccl ########
        with tf.variable_scope(tf.get_variable_scope()):
            global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                               initializer=tf.constant_initializer(0), trainable=False)
            model = build_model(graph, input_data, eval_data)
    return model, model.train_op

def train_batch(sess, num_gpu, _model, train_op):
    if num_gpu >= 1:
        model = _model
        fetche = [train_op]
        if True:
           fetche = fetche + [model.deep_loss, model.log_loss, model.l2_loss, model.train_preds, model.labels]
           _, _deeploss_, _log_loss_, _l2_loss_, _preds_, _train_labels= sess.run(fetches=fetche)
           _loss_ = _deeploss_
    else:
        pass 
    return _loss_, _log_loss_, _l2_loss_, _preds_, _train_labels

def create_dirs(dir):
    """create dir recursively as needed."""
    if not os.path.exists(dir):
        os.makedirs(dir)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--over_dump", default=False,
                        help="whether to enable overflow")
    parser.add_argument("--over_dump_path", default="./",
                        help="path to save overflow dump files")
    parser.add_argument("--data_path", default="./data",
                        help="path of dataset")
    parser.add_argument("--ckpt_path", default="./model",
                        help="path of ckpt")
    parser.add_argument("--train_size", default= config.train_size,
                        help="size of train data")
    parser.add_argument("--display_step", default= config.display_step,
                        help="display step")
    parser.add_argument('--precision_mode', default='allow_mix_precision',
                        help='allow_fp32_to_fp16/force_fp16/ '
                             'must_keep_origin_dtype/allow_mix_precision.')
    args = parser.parse_args()
    '''args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")'''
    return args


if __name__ == '__main__':
    display_step = config.display_step
    #os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

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

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    args = parse_args()
   
    #""" 
    # for npu
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    #modify enable autotune 
    if "autotune" in os.environ:
        
        
        print ("autotune value is " + os.environ["autotune"])
            
        if os.environ["autotune"] == "True":
            
            print ("autotune is set !")
            custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    #autotune end

    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = True ##True getNext false在host侧
    #custom_op.parameter_map["mix_compile_mode"].b = True  #开启混合计算，根据实际情况配置
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    custom_op.parameter_map["hcom_parallel"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = config.iterations_per_loop
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_performance")
    custom_op.parameter_map["optypelist_for_implmode"].s = tf.compat.as_bytes("UnsortedSegmentSum, GatherV2")
    if args.precision_mode == "allow_mix_precision":
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("ops_info.json")
    custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("fusion_switch.cfg")
    #aic err debug
   # custom_op.parameter_map["enable_exception_dump"].i = 1 
   # custom_op.parameter_map["op_debug_level"].i = 2 

    if args.over_dump is True:
        print("NPU overflow dump is enabled")
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    else:
        print("NPU overflow dump is disabled")


    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF


    #"""

    # for op perf
    run_metadata = tf.RunMetadata()
    #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
     
    global_start_time = time.time()
    graph = tf.Graph()
    with graph.as_default():
        train_dataset = input_fn_tfrecord(config.train_tag, int(batch_size), int(config.line_per_sample),
                                          num_epochs=config.n_epoches, perform_shuffle=True)
        eval_dataset = input_fn_tfrecord(config.eval_tag, int(config.eval_batch_size), int(config.line_per_sample),
                                          num_epochs=None, perform_shuffle=True)

        print('train batch size:', int(batch_size) * num_gpu / config.line_per_sample)
        if mode == 'train' and rank_size > 1:
            rank_size = int(os.getenv('RANK_SIZE'))
            #rank_id = int(os.getenv('DEVICE_INDEX'))
            rank_id = int(os.getenv('ASCEND_DEVICE_ID'))
            
            ############## for hccl ###################
            #rank_id = get_rank_id()
            #############  for hccl ###################

            print("ranksize = %d, rankid = %d" % (rank_size, rank_id))
            train_dataset = train_dataset.shard(rank_size, rank_id)

        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        eval_iterator = eval_dataset.make_initializable_iterator()
        eval_next_element = eval_iterator.get_next()

    if num_gpu >= 1:
        model, opt = build_graph(graph, next_element, eval_next_element,rank_id)

    with tf.Session(graph=graph, config=sess_config) as sess:
        #writer = tf.summary.FileWriter(config.writer_path, sess.graph)
        #####
        sess.run(iterator.initializer)
        sess.run(eval_iterator.initializer)
        sess.run(tf.global_variables_initializer())

       ##################### for hccl #################
        if mode == 'train' and rank_size > 1:
            npu_int = npu_ops.initialize_system()
            npu_shutdown = npu_ops.shutdown_system()
            sess.run(npu_int)
            input = tf.trainable_variables()
            bcast_global_variables_op = hccl_ops.broadcast(input, 0)
            sess.run(bcast_global_variables_op)

        #################### for hccl #################



        saver = tf.train.Saver()
        ####大小循环下沉方式
        train_op_sess = util.set_iteration_per_loop(sess, opt, config.iterations_per_loop)

        print('model initialized')

        if mode == 'train':
            start_time = time.time()
            #est_epoch_batches = int((train_para['train_per_epoch'] + train_para['batch_size']*num_gpu - 1) / (train_para['batch_size']*num_gpu))
            est_epoch_batches = int((train_para['train_per_epoch']) / (train_para['batch_size']*num_gpu))
            est_tot_batches = train_para['n_epoch'] * est_epoch_batches

            ###if use per_loop please use next
            est_epoch_batches = int((est_epoch_batches / config.iterations_per_loop))
            est_tot_batches = train_para['n_epoch'] * est_epoch_batches * config.iterations_per_loop

            print("est_epoch_batches====", est_epoch_batches)
            print("est_tot_batches====", est_tot_batches)

            _epoch = 1
            train_finished = False
            while _epoch < train_para['n_epoch'] + 1 and not train_finished:
                _epoch_start_time = time.time()
                epoch_loss = []
                epoch_labels = []
                epoch_preds = []
                epoch_auc = -1
                epoch_finished = False
                epoch_sample_num = 0
                epoch_finished_batches = 0
                cnt = 0
                #saver = tf.train.Saver()
                #saver.save(sess, Base_path + 'model/%s' % tag,
                #                        global_step=cnt, latest_filename='%s-checkpoint' % tag)
                while not epoch_finished and not train_finished:
                    try:
                        epoch_finished_batches += 1
                        start_time_test = time.time()
                        _loss, _log_loss, _l2_loss, p, _labels = train_batch(sess, num_gpu, model, train_op_sess)
                        end_time_test = time.time()
                        #print("time =================", (end_time_test - start_time_test))

                        epoch_loss.append(_loss)
                        epoch_labels.extend(_labels)
                        epoch_preds.extend(p)
                        epoch_sample_num += _labels.shape[0]
                        
                        dt = end_time_test - start_time_test
                        fps=train_para['batch_size'] * rank_size * config.iterations_per_loop/dt

                        if epoch_finished_batches % (display_step / num_gpu) == 0: # print step
                            if _epoch:
                                print("================epoch_finished_batches", epoch_finished_batches, display_step, num_gpu)
                                avg_loss = np.array(epoch_loss).mean()
                                epoch_auc = roc_auc_score(epoch_labels, epoch_preds)
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                epoch_labels = []
                                epoch_preds = []
                                epoch_loss = []
                                print('epoch %3d/%3d - batch %5d: loss = %f, auc = %f, device_id = %d | elapsed : %s, ETA : %s, fps : %f' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches, avg_loss, epoch_auc, rank_id,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta)), fps))
                                print("est_epoch_batches =====", est_epoch_batches)
                                avg_loss = 0
                            else:
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                print('epoch %3d/%3d - batch %5d: | elapsed : %s, ETA : %s' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))

                        #print("=============================", epoch_finished_batches %(20* est_epoch_batches))
                        if (epoch_finished_batches % (1 * est_epoch_batches) == 0) or epoch_finished:
                            epoch_finished = True
                            print('epoch %d train time = %.3f sec, #train sample = %d' %
                                  (_epoch, time.time() - _epoch_start_time, epoch_sample_num*config.iterations_per_loop ))

                            # ======== comment this block if no testset available ========
                            print("== starting evaluate == ")
                            evaluate_time_start = time.time()
                            eval_labels, eval_preds = evaluate(sess, model)
                            evaluate_time_end = time.time()
                            print("evaluate time =====", (evaluate_time_end - evaluate_time_start))
                            eval_auc = metric(log_file, epoch_auc, eval_labels, eval_preds)
                            print("== finished evaluate == ")
                            # ============================================================
                            print('epoch %d total time = %s' %
                                  (_epoch, str(time.time() - _epoch_start_time)))

                            if eval_auc >= metric_best:
                                metric_best = eval_auc
                                metric_best_epoch = _epoch
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
                saver.save(sess, Base_path + 'model/%s' % tag,
                                        global_step=epoch_finished_batches, latest_filename='%s-checkpoint' % tag)
            #writer.close()

        elif mode == 'test':
            pass
        tf.train.write_graph(sess.graph, config.graph_path, 'widedeep16_graph.pbtxt', as_text=True)
        ############### for hccl $^?################
        if mode == 'train' and rank_size > 1:
            sess.run(npu_shutdown)
         ############### for hccl $^?################
        sess.close()