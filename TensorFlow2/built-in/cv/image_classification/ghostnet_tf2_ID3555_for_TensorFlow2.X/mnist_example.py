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

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import time
import npu_device
import ast
# Import our GhostNet Model
from ghost_model import GhostNet

# Set seed for reproducability
seed = 1234
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=5, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./datasets", help="data path")
    parser.add_argument('--log_steps', type=int, default=40, help='log steps')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')

    parser.add_argument('--static', action='store_true', default=False, help="static")
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

args = get_args()
npu_config(args)
DATA_PATH = args.data_path
BATCH_SIZE = args.batch_size
TRAIN_EPOCHS = args.train_epochs
MODEL_DIR = args.model_dir
LOG_STEPS = args.log_steps

# ## Load in MNIST Dataset
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True,
                                         as_supervised=True, with_info=True, download=False, data_dir=DATA_PATH)


def normalize_img(image, label):
    """
    Normalizes images: `uint8` -> `float32`.
    """
    return tf.cast(image, tf.float32) / 255., label


# ## Prepare dataloaders
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
if args.static:
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
else:
    ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if args.static:
    ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
else:
    ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


# Specify number of classes for GhostNet (10 for MNIST)
model = GhostNet(10)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])


# ## Train model on MNIST
model.fit(ds_train, epochs=TRAIN_EPOCHS, validation_data=ds_test, callbacks=[TimeHistory(BATCH_SIZE, LOG_STEPS)], verbose=2)
model.save_weights(filepath=os.path.join(MODEL_DIR, 'tf_model'), save_format='tf')
