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
'''
Descripttion: train STAMP model
Author: Ziyao Geng
Date: 2020-10-25 09:27:23
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-27 10:39:34
'''
import npu_device
import ast

from time import time
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import os

from model import STAMP
from modules import *
from evaluate import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='./',
                        help="""directory to data""")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=30, type=int,
                        help="""epochs""")
    parser.add_argument('--steps_per_epoch', default=50, type=int,
                        help="""Eval batch size""")
    parser.add_argument('--learning_rate', default=0.005, type=float,
                        help="""The value of learning_rate""")
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,
                        help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                        help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10",
                        help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,
                        help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str, help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str, help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str, help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                        help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                        help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,
                        help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,
                        help='fusion_off file name, default is fusion_switch.cfg')
    args = parser.parse_args()
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
        npu_device.global_options().precision_mode = args.precision_mode
        if args.use_mixlist and args.precision_mode == 'allow_mix_precision':
            npu_device.global_options().modify_mixlist = "../configs/" + args.mixlist_file
        if args.fusion_off_flag:
            npu_device.global_options().fusion_switch_file = "../configs/" + args.fusion_off_file
        npu_device.open().as_default()
    print("Npu init")
    npu_config()
    # args, unknown_args = parser.parse_known_args()
    return args


args = parse_args()
data_dir = args.data_path

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
            self.start_time = time()
        self.epoch_start = time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time()

    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time()
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
        epoch_run_time = time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # file = '../dataset/Diginetica/train-item-views.csv'
    dirname = "train-item-views.csv"
    file = os.path.join(data_dir, dirname)
    maxlen = 8

    embed_dim = 100
    K = 20

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    # ========================== Create dataset =======================
    feature_columns, behavior_list, item_pooling, train, val, test = create_diginetica_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    # ============================Build Model==========================
    model = STAMP(feature_columns, behavior_list, item_pooling, maxlen)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/sas_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    # CrossEntropy()
    # tf.losses.SparseCategoricalCrossentropy()
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=learning_rate))

    for epoch in range(epochs):
        # ===========================Fit==============================
        t1 = time()
        model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y),
            epochs=1,
            callbacks=[TimeHistory(128,50)],
            # callbacks=[tensorboard, checkpoint],
            batch_size=batch_size,
            verbose=2,
            steps_per_epoch=steps_per_epoch,
        )
    # model.save_weights(filepath="STAMP", save_format="tf")
    # ===========================Test==============================
    #    t2 = time()
    #    hit_rate, mrr = evaluate_model(model, test, K)
    #    print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, '
    #          % (epoch, t2 - t1, time() - t2, hit_rate, mrr))
