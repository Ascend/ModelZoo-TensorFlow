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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig

tf.get_logger().setLevel('ERROR')
import argparse
import ast
import npu_device
import time
import npu_convert_dropout


def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=2, help="epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./data", help="data path")
    parser.add_argument('--log_steps', type=int, default=60, help='log steps')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
    parser.add_argument('--perf', action='store_true', default=False, help="test performanc, set train data size 800")
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

RoBERTa_PATH = os.path.join(DATA_PATH, 'pretrain_model')
tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_PATH)

MAX_SEQUENCE_LENGTH = 384

if args.perf:
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), nrows=800)
else:
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
# df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
# df_sub = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
print('train shape =', df_train.shape)
# print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        # input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        # input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks]
    
    input_ids_q, input_masks_q = return_id(
        title + ' ' + question, None, 'longest_first', max_sequence_length)
    
    input_ids_a, input_masks_a = return_id(
        answer, None, 'longest_first', max_sequence_length)
    
    return [input_ids_q, input_masks_q,
            input_ids_a, input_masks_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q = [], []
    input_ids_a, input_masks_a = [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, ids_a, masks_a = \
        _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        # input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        # input_segments_a.append(segments_a)
    if args.static:
        size = len(input_ids_q) // BATCH_SIZE * BATCH_SIZE
        return [np.asarray(input_ids_q[:size], dtype=np.int32), 
                np.asarray(input_masks_q[:size], dtype=np.int32), 
                np.asarray(input_ids_a[:size], dtype=np.int32),
                np.asarray(input_masks_a[:size], dtype=np.int32)]
    else:
        return [np.asarray(input_ids_q, dtype=np.int32), 
                np.asarray(input_masks_q, dtype=np.int32), 
                np.asarray(input_ids_a, dtype=np.int32),
                np.asarray(input_masks_a, dtype=np.int32)]


def compute_output_arrays(df, columns):
    if args.static:
        size = len(df[columns]) // BATCH_SIZE * BATCH_SIZE
        return np.asarray(df[columns][:size])
    return np.asarray(df[columns])


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = RobertaConfig()
    config.vocab_size = 50265
    config.max_position_embeddings = 514
    config.type_vocab_size = 1
    print(config)
    config.output_hidden_states = False # Set to True to obtain hidden states
    roberta_model = TFRobertaModel.from_pretrained(os.path.join(RoBERTa_PATH, 'tf_model.h5'), config=config)

    # RoBERTa version of embeddings
    q_embedding = roberta_model(q_id, attention_mask=q_mask)[0]
    a_embedding = roberta_model(a_id, attention_mask=a_mask)[0]
    
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
    x = tf.keras.layers.Concatenate()([q, a])
    
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, a_id, a_mask], outputs=x)
    
    return model


outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
model.fit(inputs, outputs, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, callbacks=[TimeHistory(BATCH_SIZE, LOG_STEPS)], verbose=2)
model.save_weights(filepath=os.path.join(MODEL_DIR, 'tf_model'), save_format='tf')




