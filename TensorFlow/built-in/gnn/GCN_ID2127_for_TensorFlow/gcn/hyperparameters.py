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
# Author: Tao Wu (taowu1@huawei.com)

""" hyperparameters.py """

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('device', 'gpu', 'choose device among {cpu, gpu, npu}')
flags.DEFINE_integer('device_id', 0, 'device id for gpu')
flags.DEFINE_string('data_path', './data', 'data path')
flags.DEFINE_string('out_path', './results', 'output path')
flags.DEFINE_string('ckpt_file', 'gcn_checkpoint', 'checkpoint name')
flags.DEFINE_string('pb_file', 'constant_graph.pb', 'protobuf name')
flags.DEFINE_integer('seed', None, 'random seed')

flags.DEFINE_bool('cora_full', False, 'use cora_full or cora')
flags.DEFINE_bool('shuffle', False, 'shuffle data')
flags.DEFINE_bool('take_subgraphs', False, 'take largest connected subgraphs')
flags.DEFINE_integer('hidden_dim', 16, 'hidden dimension of gcn')
flags.DEFINE_float('l2_regularizer', 5e-4, 'l2 regularizer weight')
flags.DEFINE_float('keep_prob', 0.5, 'dropout keep probability')
flags.DEFINE_integer('num_epochs', 200, 'number of training epochs')
flags.DEFINE_integer('display_per_epochs', 1, 'display progress per epochs')
flags.DEFINE_integer('train_size', None, 'training dataset size')
flags.DEFINE_integer('valid_size', 500, 'validation dataset size')
flags.DEFINE_integer('test_size', 1000, 'test dataset size')
flags.DEFINE_integer('min_train_samples', 20, 'minimal training samples per class')
flags.DEFINE_integer('min_valid_samples', 30, 'minimal validation samples per class')
flags.DEFINE_float('learning_rate', 1e-2, 'learning rate for optimizer')
flags.DEFINE_integer('patience', 20, 'early stopping rule')
flags.DEFINE_bool('sparse_input', False, 'sparse input features')
flags.DEFINE_bool('sparse_adj', False, 'sparse adjacency')

flags.DEFINE_bool('profiling_mode', False, 'Activate profiling mode')
flags.DEFINE_string('profiling_options', 'task_trace', 'Type of the profiling')

flags.DEFINE_bool('over_dump', False, 'if or not over detection, default is False')
flags.DEFINE_bool('data_dump_flag', False, 'data dump flag, default is False')
flags.DEFINE_integer('data_dump_step', 10, 'data dump step, default is 10')

assert FLAGS.device in ['cpu', 'gpu', 'npu'], "Unknown device type."
assert FLAGS.num_epochs >= 1, "num_epochs must be at least 1."
assert 0 < FLAGS.keep_prob <= 1, "keep_prob must be a positive scalar."
