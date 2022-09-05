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
import tensorflow as tf
import numpy as np
import ast
import npu_device
import argparse
import time
import os

from dataset import get_dataset, prepare_dataset
from model import get_model

def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./", help="data_path")
    parser.add_argument('--log_steps', type=int, default=200, help='Learning rate for training')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
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

args = get_args()
npu_config(args)

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

dataset = get_dataset("fr-en", args.data_path)

print("Dataset loaded. Length:", len(dataset), "lines")
train_dataset = dataset[0:100000]
batch_size = args.batch_size
if args.rank_size !=1:
    train_data = np.split(np.array(train_dataset), args.rank_size)
    train_data = train_data[args.device_id]

    train_labels = np.split(np.array(train_dataset), args.rank_size)
    train_labels = train_labels[args.device_id]

    batch_size = batch_size // args.rank_size
train_data_size = len(train_dataset) // batch_size * batch_size

train_dataset = train_dataset[0:train_data_size]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128,
  BATCH_SIZE = batch_size,
  MAX_WIN_SIZE = 20

)
transformer_model.build(input_shape = (20,))
if args.rank_size !=1:
    optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.Adam())
else:
    optimizer = tf.keras.optimizers.Adam()
transformer_model.compile(
  optimizer = optimizer,
  loss = [
    "sparse_categorical_crossentropy"
  ],
  metrics = [
    "accuracy"
  ]
)
if args.rank_size !=1:
    training_vars = transformer_model.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank=0)

transformer_model.summary()

x = [np.array(encoder_input), np.array(decoder_input)]
y = np.array(decoder_output)

name = "transformer"
checkpoint_filepath = "./logs/transformer_ep-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir = "logs/{}".format(name)
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath = checkpoint_filepath,
  monitor = "val_accuracy",
  mode = "max",
  save_weights_only = True,
  save_best_only = True,
  verbose = True
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
  monitor = "val_accuracy",
  mode = "max",
  patience = 2,
  min_delta = 0.001,
  verbose = True
)

transformer_model.fit(
  x,
  y,
  epochs = args.train_epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  callbacks=[
    TimeHistory(args.batch_size, args.log_steps//args.rank_size)
  ],
  verbose=2,
)
