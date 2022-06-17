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
# import npu_device
# npu_device.open().as_default()

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from npu_device.compat.v1.npu_init import *
import npu_device 
npu_device.compat.enable_v1()

import os
import argparse
# import numpy as np
# from tqdm import tqdm
# import pandas as pd
import joblib
# from collections import OrderedDict

# import keras
#from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
# from keras.preprocessing.image import ImageDataGenerator
# from keras.regularizers import l2
from keras import backend as K
# from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils

from utils import *
from wide_resnet import *
from cosine_annealing import *
from dataset import Cifar10ImageDataGenerator
import datetime
import numpy as np
import time
import os
import argparse
import ast

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes("16 * 1024 * 1024 * 1024")
custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes("15 * 1024 * 1024 * 1024")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)


starttime = datetime.datetime.now()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,
                        help='the path to save over dump data')
    parser.add_argument('--data_path', default="", type=str,
                        help='the path of dataset')
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
    parser.add_argument('--auto_tune', dest="auto_tune", type=ast.literal_eval,
                        help='auto_tune flag')
    parser.add_argument('--static', default=0, type=int)
    args = parser.parse_args()

    return args



class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
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


args = parse_args()
#===============================NPU Migration=========================================
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
                          "L2":"on", \
                          "aic_metrics":"PipeUtilization",\
                          "fp_point":"", \
                          "bp_point":""}'
      npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode = args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist="../configs/"+args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file="../configs/"+args.fusion_off_file
  if args.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()
#===============================NPU Migration=========================================

def main():
    args = parse_args()
    npu_config()
    if args.name is None:
        args.name = 'WideResNet%s-%s' %(args.depth, args.width)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    endtime = datetime.datetime.now()
    TOTLE_TIME = (endtime - starttime).seconds
    print("TOTLE_TIME : ", TOTLE_TIME)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    model = WideResNet(args.depth, args.width, num_classes=10)
    model.compile(loss='categorical_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    datagen = Cifar10ImageDataGenerator(args)

    x_test = datagen.standardize(x_test)

    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    if args.static==1:
        x_train = np.array(x_train[:49920])
        y_train = np.array(y_train[:49920])
        x_test = np.array(x_test[:9984])
        y_test = np.array(y_test[:9984])
        print("x_train:",np.array(x_train).shape,flush=True)
        print("y_train:",np.array(y_train).shape,flush=True)
        print('=========================test Parameters =======================')
    
    '''
        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, x_test))
        .shuffle(args.batch_size)
        .batch(args.batch_size, drop_remainder=True))
    else:
        train_ds = (tf.data.Dataset.from_tensor_slices((x_train, x_test))
        .shuffle(args.batch_size)
        .batch(args.batch_size, drop_remainder=False))
    
    callbacks = [
        ModelCheckpoint('models/%s/model.hdf5'%args.name, verbose=1, save_best_only=True),
        CSVLogger('models/%s/log.csv'%args.name),
        CosineAnnealingScheduler(T_max=args.epochs, eta_max=0.05, eta_min=4e-4),
        TimeHistory(args.batch_size,195)
    ]
    '''
    callbacks = [TimeHistory(args.batch_size,195)]
    '''
    model.fit_generator(datagen.flow(train_ds, batch_size=args.batch_size),
                        steps_per_epoch=len(x_train)//args.batch_size,
                        validation_data=(x_test, y_test),
                        epochs=args.epochs,
                        verbose=2,
                        callbacks=callbacks,
                        )
    '''
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=len(x_train)//args.batch_size,
                        validation_data=(x_test, y_test),
                        epochs=args.epochs,
                        verbose=2,
                        callbacks=callbacks,
                        )
   
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    main()
    sess.close()

