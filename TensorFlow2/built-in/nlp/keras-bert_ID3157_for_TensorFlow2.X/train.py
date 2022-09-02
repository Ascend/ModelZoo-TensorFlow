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

import npu_device
import argparse
import time
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./', help='data dir')
parser.add_argument('--train_epochs', type=int, default=20, help='Training epoch')
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
parser.add_argument('--log_steps', type=float, default=50, help='Learning rate for training')

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

import numpy as np
import tensorflow as tf
import os
import codecs
import keras
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tqdm import tqdm
import tensorflow_addons as tfa


SEQ_LEN = 32
BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.train_epochs
LR = 1e-4
LOG_STEPS= FLAGS.log_steps


pretrained_path = os.path.join(FLAGS.data_path, 'uncased_L-12_H-768_A-12')
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
	for line in reader:
		token = line.strip()
		token_dict[token] = len(token_dict)

model = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,trainable=True,seq_len=SEQ_LEN)

'''
# Raw data load

dataset = os.path.join(FLAGS.data_path, 'aclImdb')
# dataset = tf.keras.utils.get_file(fname="aclImdb.tar.gz",origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",extract=True,)
tokenizer = Tokenizer(token_dict)

def load_data(path, prefix=None):
    global tokenizer
    indices, sentiments = [],[]
    for folder, sentiment in (('neg',0),):
        folder = os.path.join(path,folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name),'r',encoding='UTF-8') as reader:
                text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            sentiments.append(sentiment)
    items = list(zip(indices,sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    if prefix:
        np.save(f"{prefix}_indices.npy", indices)
        np.save(f"{prefix}_sentiments.npy", np.array(sentiments))
    return [indices, np.zeros_like(indices)], np.array(sentiments)

train_path = os.path.join(os.path.dirname(dataset), 'aclImdb', 'train')
#test_path = os.path.join(os.path.dirname(dataset), 'aclImdb', 'test')

train_x, train_y = load_data(train_path)
# test_x, test_y = load_data(test_path)
'''

# load fixed data files(indices.npy and sentiments.npy)
def read_data_from_npy(indices_path, sentiments_path):
    indices = np.load(indices_path)
    sentiments = np.load(sentiments_path)
    return [indices, np.zeros_like(indices)], sentiments

indices_path = os.path.join(FLAGS.data_path, 'train_indices.npy')
sentiments_path = os.path.join(FLAGS.data_path, 'train_sentiments.npy')
train_x, train_y = read_data_from_npy(indices_path, sentiments_path)

inputs =model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = tf.keras.layers.Dense(units=2, activation='softmax')(dense)

model = keras.models.Model(inputs, outputs)
model.compile(tfa.optimizers.RectifiedAdam(learning_rate=LR),loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'],)

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
model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[TimeHistory(BATCH_SIZE,LOG_STEPS)], verbose=2)

model.save_weights(filepath="tf_model",save_format="tf")
