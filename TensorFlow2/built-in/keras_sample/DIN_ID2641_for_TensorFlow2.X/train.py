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
Created on Oct 23, 2020

train DIN model

@author: Ziyao Geng
"""

import npu_device

import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from model import DIN
from utils import *

import os
import ast
import time
import numpy as np
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

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        #self.opt = optimizer
        self.start_time = None

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size
            print(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d'%(elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps),flush=True)
            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()
    def on_batch_end(self, batch, logs={}):
        if batch % args.log_steps == 0:
            loss = logs.get('loss')
            dura = time.time() - self.start
            if dura < 10:
                self.epoch_perf.append(dura)
            print('step:%d ,loss: %f ,time:%f'%(batch, loss, dura), flush=True)
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_perf = []
        self.epochstart = time.time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time.time() - self.epochstart
        print('epoch_duration: ', duration)
        if epoch != 0:
            self.perf.append(np.mean(self.epoch_perf))
    def on_train_begin(self, logs={}):
        print('params: ', self.params)
        self.perf = []
    def on_train_end(self, logs={}):
        print('imgs/s: %.2f'%(self.batch_size / np.mean(self.perf)))

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    file = 'remap.pkl'
    file = os.path.join(args.data_path, file)
    maxlen = 20
    
    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = args.batch_size
    epochs = args.epochs
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_amazon_electronic_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    print('=========================train Parameters =======================')
    if args.static==1:
        train_X = [np.array(train_X[0][:2220032]),np.array(train_X[1][:2220032]), np.array(train_X[2][:2220032]), np.array(train_X[3][:2220032])]
        train_y = np.array(train_y[:2220032])
        val_X = [np.array(val_X[0][:380928]),np.array(val_X[1][:380928]), np.array(val_X[2][:380928]), np.array(val_X[3][:380928])]
        val_y = np.array(val_y[:380928])
        test_X = [np.array(test_X[0][:380928]),np.array(test_X[1][:380928]), np.array(test_X[2][:380928]), np.array(test_X[3][:380928])]
        test_y = np.array(test_y[:380928])
    print('=========================test Parameters =======================')
    
    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation, 
        ffn_activation, maxlen, dnn_dropout)
    model.summary()
    #logger = LossHistory(batch_size)
    logger = TimeHistory(batch_size,100)
    # ============================model checkpoint======================
    # check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        callbacks=logger,
        validation_data=(val_X, val_y),
        batch_size=batch_size,
        verbose=2
    )
    save_ckpt = os.path.join(args.ckpt_save_path, "checkpoint/tf_model")
    model.save_weights(filepath=save_ckpt, save_format="tf")
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size, verbose=2)[1])
