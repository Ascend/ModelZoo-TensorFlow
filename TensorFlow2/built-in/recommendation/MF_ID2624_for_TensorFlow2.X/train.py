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
Created on August 31, 2020

train model

@author: Ziyao Geng
"""

import os
import npu_device
import ast
import argparse
#===============================NPU Migration=========================================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                    help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                    help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",
                    help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
parser.add_argument('--data_path', default='./',help="""directory to data""")
parser.add_argument('--batch_size', default=32, type=int,help="""batch size for 1p""")
parser.add_argument('--epochs', default=2, type=int,help="""epochs""")
parser.add_argument('--static', default=1, type=int,help="""static shape""")
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
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist="../configs/"+args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file="../configs/"+args.fusion_off_file
  npu_device.open().as_default()
#===============================NPU Migration=========================================

npu_config()

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import MF
from utils import *
from time import time
import os
import warnings
from sys import argv
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    file_path = args.data_path
    epochs = args.epochs
    static = args.static
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = file_path + '/ratings.dat'
    test_size = 0.2

    latent_dim = 32
    # use bias
    use_bias = True

    learning_rate = 0.001
    batch_size = args.batch_size
    total_steps = 797696 // batch_size

    # ========================== Create dataset =======================
    feature_columns, train, test = create_explicit_ml_1m_dataset(file, latent_dim, test_size)
    train_X, train_y = train
    test_X, test_y = test

    dataset = tf.data.Dataset.from_tensor_slices(((train_X[0], train_X[1]), train_y))
    if static:
        dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=64)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False, num_parallel_calls=64)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 64
    dataset = dataset.with_options(options)
    dataset = dataset.cache()
    # ============================Build Model==========================
    model = MF(feature_columns, use_bias)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/mf_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    logger = LossHistory()
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),
                    metrics=['mse'])
    # ==============================Fit==============================
    history = model.fit(
            dataset,
            epochs=epochs,
            verbose=2,
            callbacks=[logger]
            )
    # model.save_weights(filepath="model",save_format="tf")
    # ===========================Test==============================
    print('test rmse: %.2f' % np.sqrt(model.evaluate(test_X, test_y, verbose=0)[1]))