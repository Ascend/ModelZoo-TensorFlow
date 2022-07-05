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
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""
import os
import ast
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import npu_convert_dropout
import npu_device
import time

"""
## Prepare the data
"""
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='./',
                        help="""directory to data""")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=50, type=int,
                        help="""epochs""")
    parser.add_argument("--log_steps", default=50, type=int,
                        help="TimeHis log Step.")
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
    parser.add_argument('--auto_tune', dest="auto_tune", type=ast.literal_eval,
                        help='auto_tune flag')
    parser.add_argument('--static', dest="static", type=ast.literal_eval,
                        help="""judgement dynamic or static shape""")
    '''
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    '''
    return parser.parse_args()

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
npu_config()
data_dir = args.data_path

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

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

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
def load_data(data_dir):
  origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  dirname = 'mnist.npz'
  path = os.path.join(data_dir, dirname)
  with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data(data_dir)

#if args.eval_static:
#  x_train = x_train[:59968]
#  y_train = y_train[:59968]
#  x_test = x_test[:9984]
#  y_test = y_test[:9984]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

#batch_size = 128
batch_size = args.batch_size
#epochs = 15
epochs = args.epochs

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
if args.static:
    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(args.batch_size)
        .batch(args.batch_size, drop_remainder=args.static)
    )
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, callbacks=[TimeHistory(batch_size,844)], verbose=2)
else:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[TimeHistory(batch_size,844)], verbose=2)
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save_weights(filepath="mnist_convnet", save_format="tf")


## Evaluate the trained model


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
