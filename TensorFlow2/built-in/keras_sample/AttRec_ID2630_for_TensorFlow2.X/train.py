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
Created on Nov 11, 2020

train AttRec model

@author: Ziyao Geng
"""

import npu_device

import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from tensorflow.keras.optimizers import Adam

from model import AttRec
from modules import *
from evaluate import *
from utils import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='', help="""directory to data""")
    parser.add_argument('--ckpt_save_path', default='', help="""directory to ckpt""")
    parser.add_argument('--batch_size', default=32, type=int, help="""batch size for 1p""")
    parser.add_argument('--epochs', default=3, type=int, help="""epochs""")
    parser.add_argument('--log_steps', default=1, type=int, help="""log frequency""")
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--static', default=0, type=int, help="""static""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


args = parse_args()
def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
    npu_device.global_options().dump_config.enable_dump_debug = True
    npu_device.global_options().dump_config.dump_path = args.over_dump_path
    npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
    npu_device.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"", \
                        "bp_point":""}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  npu_device.open().as_default()

npu_config()

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
    def on_batch_begin(self, batch, logs={}):
        self.start = time()
    def on_batch_end(self, batch, logs={}):
        if batch % args.log_steps == 0:
            loss = logs.get('loss')
            dura = time() - self.start
            if dura < 10:
                self.epoch_perf.append(dura)
            print('step:%d ,loss: %f ,time:%f'%(batch, loss, dura), flush=True)
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_perf = []
        self.epochstart = time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time() - self.epochstart
        print('epoch_duration: ', duration)
        self.perf.append(np.mean(self.epoch_perf))
    def on_train_begin(self, logs={}):
        print('params: ', self.params)
        self.perf = []
    def on_train_end(self, logs={}):
        print('imgs/s: %.2f'%(self.batch_size / np.mean(self.perf)))


if __name__ == '__main__':

    # ========================= Hyper Parameters =======================
    file = 'ratings.dat'
    file = os.path.join(args.data_path, file)
    print(file)
    trans_score = 1
    maxlen = 5
    
    embed_dim = 100
    embed_reg = 1e-6  # 1e-6
    gamma = 0.5
    mode = 'inner'  # 'inner' or 'dist'
    w = 0.5
    K = 10

    learning_rate = 0.001
    epochs = args.epochs
    batch_size = args.batch_size
    # ========================== Create dataset =======================
    feature_columns, train, val, test = create_implicit_ml_1m_dataset(file, trans_score, embed_dim, maxlen)
    if args.static==1:
        print('=====================[DEBUG]======================',flush=True)
        train_X = [np.array(train[0][:982016],dtype='int32'),np.array(train[1][:982016],dtype='int32'),np.array(train[2][:982016],dtype='int32'),np.array(train[3][:982016],dtype='int32')]
        val_X = [np.array(val[0][:5632],dtype='int32'),np.array(val[1][:5632],dtype='int32'),np.array(val[2][:5632],dtype='int32'),np.array(val[3][:5632],dtype='int32')]
        print(train_X[0].shape,train_X[1].shape,train_X[2].shape,train_X[3].shape,flush=True)
        print(val_X[0].shape,val_X[1].shape,val_X[2].shape,val_X[3].shape,flush=True)

        #train_X = train[:491520]
        #val_X = val[:491520]
    else:
        train_X = train
        val_X = val
    # ============================Build Model==========================
    model = AttRec(feature_columns, maxlen, mode, gamma, w, embed_reg)
    model.summary()
    logger = LossHistory(batch_size)
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            None,
            validation_data=(val_X, None),
            epochs=1,
            # callbacks=[tensorboard, checkpoint],
            callbacks=logger,
            batch_size=batch_size,
            verbose=2
            )
        save_ckpt = os.path.join(args.ckpt_save_path, "checkpoint/tf_model")
        #model.save_weights(filepath=save_ckpt, save_format="tf")
        # ===========================Test==============================
        t2 = time()
        if epoch % 5 == 0:
            hit_rate, ndcg, mrr = evaluate_model(model, test, K)
            print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, MRR = %.4f'
                  % (epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr))
            results.append([epoch, t2 - t1, time() - t2, hit_rate, ndcg, mrr])
    # ========================== Write Log ===========================
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time',
                                   'hit_rate', 'ndcg', 'mrr']).to_csv(
        'log/AttRec_log_maxlen_{}_dim_{}_K_{}_w_{}.csv'.format(maxlen, embed_dim, K, w), index=False)
