#
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
import ast
import npu_device
import os
import time
import numpy as np
import tensorflow as tf
from model import MnasNet
from keras import backend
from tensorflow.keras import utils
from keras.datasets.cifar import load_batch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default='dataset', type=str, help="directory to data")
    parser.add_argument("--epochs", default=100, type=int, help="epochs")
    parser.add_argument("--batch_size", default=2048, type=int, help="batch size")
    parser.add_argument("--log_steps", default=25, type=int, help="log steps")
    parser.add_argument('--static', action='store_true', default=False, help="static shape")

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

    ############多p参数##############
    parser.add_argument("--rank_size", default=1, type=int, help="rank size")
    parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")


    args, unknown_args = parser.parse_known_args()
    return args

def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()

args = parse_args()
npu_config(args)

def load_data(path='dataset'):
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)

input_shape = (32, 32)
batch_size = args.batch_size
nb_classes = 10
epochs = args.epochs

model = MnasNet(input_shape=input_shape+(3,), pooling='avg', nb_classes=nb_classes)
if args.rank_size !=1:
    optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.Adam())
else:
    optimizer = tf.keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if args.rank_size !=1:
    training_vars = model.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank=0)

(x_train, y_train), (x_test, y_test) = load_data(args.data_dir)

y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if args.static:
    x_train_size = x_train.shape[0] // args.batch_size * args.batch_size
    x_train = x_train[:x_train_size]
    y_train_size = y_train.shape[0] // args.batch_size * args.batch_size
    y_train = y_train[:y_train_size]
    x_test_size = x_test.shape[0] // args.batch_size * args.batch_size
    x_test = x_test[:x_test_size]
    y_test_size = y_test.shape[0] // args.batch_size * args.batch_size
    y_test = y_test[:y_test_size]

if args.rank_size !=1:
    x_train = np.split(x_train, args.rank_size)
    x_train = x_train[args.device_id]
    y_train = np.split(y_train, args.rank_size)
    y_train = y_train[args.device_id]
    batch_size = args.batch_size // args.rank_size

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=2,
          callbacks=[TimeHistory(args.batch_size, args.log_steps//args.rank_size)])
model.save_weights(filepath="checkpoint/tf_model", save_format="tf")
