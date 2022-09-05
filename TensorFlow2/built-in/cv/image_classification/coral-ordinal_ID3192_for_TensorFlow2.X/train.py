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
"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""


import npu_device
import argparse
import ast
import os
from tensorflow import keras
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./', help='data dir')
parser.add_argument('--train_epochs', type=int, default=50, help='Training epoch')
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
parser.add_argument('--log_steps', type=float, default=150, help='Learning rate for training')
parser.add_argument('--static', action='store_true', default=False, help='static input shape, default is False')

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

FLAGS, unparsed = parser.parse_known_args()


def npu_config():
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


npu_config()


import tensorflow as tf
import coral_ordinal as coral
import time
from sklearn import model_selection

data_path = FLAGS.data_path
batch_size = FLAGS.batch_size
epochs = FLAGS.train_epochs
log_steps = FLAGS.log_steps

NUM_CLASSES = 10
random_seed = 1
learning_rate = 0.05

(mnist_images, mnist_labels),(mnist_images_test, mnist_labels_test) = tf.keras.datasets.mnist.load_data(os.path.join(data_path, 'mnist.npz'))

mnist_images, mnist_images_val, mnist_labels, mnist_labels_val = model_selection.train_test_split(mnist_images,mnist_labels,test_size=5000,random_state=1)

dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images[...,tf.newaxis] / 255, tf.float32),tf.cast(mnist_labels,tf.int64)))

if FLAGS.rank_size !=1:
    dataset, batch_size = npu_device.distribute.shard_and_rebatch_dataset(dataset, batch_size)

dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=FLAGS.static)


test_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images_test[...,tf.newaxis] / 255, tf.float32),tf.cast(mnist_labels_test,tf.int64)))
test_dataset = test_dataset.batch(batch_size, drop_remainder=FLAGS.static)

val_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images_val[...,tf.newaxis] / 255, tf.float32),tf.cast(mnist_labels_val,tf.int64)))
val_dataset = val_dataset.shuffle(1000).batch(batch_size, drop_remainder=FLAGS.static)


def create_model(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28,)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(coral.CoralOrdinal(num_classes))
    return model

model = create_model(NUM_CLASSES)
model.summary()

if FLAGS.rank_size !=1:
    optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(keras.optimizers.Adam())
else:
    optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer,loss=coral.OrdinalCrossEntropy(num_classes=NUM_CLASSES),metrics=[coral.MeanAbsoluteErrorLabels()])

if FLAGS.rank_size !=1:
    training_vars = model.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank=0)

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

# Start training
model.fit(dataset, epochs=epochs, validation_data=val_dataset, callbacks=[TimeHistory(batch_size*FLAGS.rank_size, log_steps//FLAGS.rank_size)], verbose=2)


model.save_weights(filepath="../gpu_ckpt/tf_model",save_format="tf")
