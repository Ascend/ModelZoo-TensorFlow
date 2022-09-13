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
DenseNet model for Cifar10 classification.
"""
import npu_device as npu
# npu.open().as_default()

import os

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import cifar10
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

import utils
from DenseNet import DenseNet
import argparse
import time
import ast

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


def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=30, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./data", help="data_path")
    parser.add_argument('--log_steps', type=int, default=400, help='log steps')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
    parser.add_argument('--static', action='store_true', default=False, help='static input shape, default is False')
    parser.add_argument('--learning_rate', type=int, default=1e-3, help='Learning rate for training')
    parser.add_argument('--train_data_size', type=int, default=50000, help='train data size')
    parser.add_argument('--test_data_size', type=int, default=10000, help='test data size')
    #===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='precision mode')
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
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')

    args = parser.parse_args()
    return args


def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu.global_options().dump_config.enable_dump = True
        npu.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu.global_options().dump_config.enable_dump_debug = True
        npu.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu.global_options().profiling_config.profiling_options = profiling_options
    npu.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu.global_options().auto_tune_mode="RL,GA"
    npu.open().as_default()

args = get_args()
npu_config(args)
DATA_PATH = args.data_path
BATCH_SIZE = args.batch_size
TRAIN_EPOCHS = args.train_epochs
MODEL_DIR = args.model_dir
LOG_STEPS = args.log_steps
LEARNING_RATE = args.learning_rate
STATIC = args.static
TRAIN_DATA_SIZE = args.train_data_size
TEST_DATA_SIZE = args.test_data_size


# Define useful paths
results_name = 'DenseNet-BC_cifar10_L100_k12_32_FIXED'

# results_path = os.path.join(os.getcwd(), 'results', results_name)
# tb_logs = os.path.join(os.getcwd(), 'logs', results_name)
#
# try:
#     os.makedirs(results_path)
# except:
#     pass
#
# try:
#     os.makedirs(tb_logs)
# except:
#     pass

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data(os.path.join(DATA_PATH, 'cifar-10-batches-py'))
x_train = x_train[:TRAIN_DATA_SIZE]
y_train = y_train[:TRAIN_DATA_SIZE]
x_test = x_test[:TEST_DATA_SIZE]
y_test = y_test[:TEST_DATA_SIZE]
if STATIC:
    data_size_train = x_train.shape[0] // BATCH_SIZE * BATCH_SIZE
    x_train = x_train[:data_size_train]
    y_train = y_train[:data_size_train]
    data_size_test = x_test.shape[0] // BATCH_SIZE * BATCH_SIZE
    x_test = x_test[:data_size_test]
    y_test = y_test[:data_size_test]

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Change mode to channels_last
if x_train.shape[3] != 3:
    x_train = np.swapaxes(x_train, 1, 3)
    x_test = np.swapaxes(x_test, 1, 3)

# Preprocess the images
channel_means = np.mean(x_train, axis=(0,1,2))
channel_stds = np.std(x_train, axis=(0,1,2))

x_train = (x_train - channel_means) / channel_stds
x_test = (x_test - channel_means) / channel_stds

# Create the model
model = DenseNet(growth_rate=12, blocks=[32,32,32], first_num_channels=2*12,
                 dropout_p=0.2, bottleneck=4*12, compression=0.5,
                 input_shape=(32,32,3), first_conv_pool=False,
                 weight_decay=1e-4, data_format='channels_last',
                 num_classes=num_classes)

# Train the model from scratch
lr = 0.1


def schedule_fn(epoch_i):
    if epoch_i >= TRAIN_EPOCHS * 0.75:
        print('learning rate = ', lr/100)
        return lr / 100
    elif epoch_i >= TRAIN_EPOCHS * 0.5:
        print('learning rate = ', lr/10)
        return lr / 10
    else:
        print('learning rate = ', lr)
        return lr


lr_scheduler = LearningRateScheduler(schedule_fn)

sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS,
                    callbacks=[lr_scheduler, TimeHistory(BATCH_SIZE, LOG_STEPS)],
                    validation_data=(x_test, y_test), verbose=2)

model.save_weights(filepath=os.path.join(MODEL_DIR, 'tf_model'), save_format='tf')

