#! usr/bin/env python3
# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
Created on Feb 26 2017
Author: Weiping Song
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from absl import flags
from tensorflow.python.ops import rnn_cell

FLAGS = flags.FLAGS

__all__ = 'BpProcess'


class BpProcess(object):
    def __init__(self):
        if FLAGS.hidden_act == "tanh":
            self.hidden_act = self.tanh
        elif FLAGS.hidden_act == "relu":
            self.hidden_act = self.relu
        else:
            raise NotImplementedError
        self.layers = FLAGS.rnn_layers
        self.rnn_size = FLAGS.rnn_size
        self.n_items = FLAGS.n_items
        self.batch = FLAGS.predict_batch_size
        if FLAGS.classifier_type == "softmax":
            self.final_activation = self.softmax
        elif FLAGS.classifier_type == "tanh":
            self.final_activation = self.softmaxth
        else:
            raise NotImplementedError
        self.predict_state = [np.zeros([self.batch, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X), name="logits")

    def softmax(self, X):
        return tf.nn.softmax(X, name="logits")

    def tanh(self, X):
        return tf.nn.tanh(X)

    def relu(self, X):
        return tf.nn.relu(X)

    def get_features(self, preprocess):
        self.create_model()
        data_dir = FLAGS.data_dir
        input_path = os.path.join(data_dir, "input")
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        rnn_path = os.path.join(data_dir, "rnn_state")
        if not os.path.exists(rnn_path):
            os.makedirs(rnn_path)
        batch_size = self.batch
        session_key = 'SessionId'
        item_key = 'ItemId'
        time_key = 'Time'
        train_data = pd.read_csv(os.path.join(data_dir, "rsc15_train_full.txt"), sep='\t',
                                 dtype={'ItemId': np.int64})
        test_data = pd.read_csv(os.path.join(data_dir, "rsc15_test.txt"), sep='\t', dtype={'ItemId': np.int64})

        itemids = train_data[item_key].unique()
        itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
        test_data.sort([session_key, time_key], inplace=True)
        offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
        if len(offset_sessions) - 1 < batch_size:
            batch_size = len(offset_sessions) - 1
        iters = np.arange(batch_size).astype(np.int32)
        maxiter = iters.max()
        start = offset_sessions[iters]
        end = offset_sessions[iters + 1]
        in_idx = np.zeros(batch_size, dtype=np.int32)
        x = 0
        while preprocess:
            valid_mask = iters >= 0
            if valid_mask.sum() == 0:
                break
            start_valid = start[valid_mask]
            minlen = (end[valid_mask] - start_valid).min()
            in_idx[valid_mask] = test_data[item_key].values[start_valid]
            for i in range(minlen - 1):
                if not self.predict:
                    self.current_session = np.ones(self.batch) * -1
                    self.predict = True
                session_change = np.arange(self.batch)[iters != self.current_session]
                if len(session_change) > 0:  # change internal states with session changes
                    for index in range(self.layers):
                        self.predict_state[index][session_change] = 0.0
                    self.current_session = iters.copy()

                in_idxs = itemidmap[in_idx]
                in_idxs = in_idxs.values.astype(np.int32)
                feed_dict = {self.X: in_idxs}
                for idx in range(self.layers):
                    feed_dict[self.state[idx]] = self.predict_state[idx]
                input_idx = 0
                for key in feed_dict.keys():
                    if input_idx == 0:
                        feed_dict[key].tofile("%s/input_%05d_%05d.bin" % (input_path, x, i))
                        input_idx = 1
                    else:
                        feed_dict[key].tofile("%s/rnn_state_%05d_%05d.bin" % (input_path, x, i))
                        input_idx = 0
            start = start + minlen - 1
            mask = np.arange(len(iters))[valid_mask & (end - start <= 1)]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(offset_sessions) - 1:
                    iters[idx] = -1
                else:
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[maxiter]
                    end[idx] = offset_sessions[maxiter + 1]
            x += 1

    def create_model(self):
        self.X = tf.placeholder(tf.int32, [self.batch], name='input')
        self.state = [tf.placeholder(tf.float32, [self.batch, self.rnn_size], name='rnn_state')
                      for _ in range(self.layers)]
        with tf.variable_scope('gru_layer'):
            sigma = np.sqrt(6.0 / (self.n_items + self.rnn_size))
            initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_w = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=1)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            logits = tf.matmul(output, softmax_w, transpose_b=True) + softmax_b
        self.yhat = self.final_activation(logits)

        return self.yhat

    def calc_precision(self):
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)
        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_pre = os.path.join(output_dir, model_name)
        cut_off = 20
        batch_size = self.batch
        session_key = 'SessionId'
        item_key = 'ItemId'
        time_key = 'Time'
        train_data = pd.read_csv(os.path.join(FLAGS.data_dir, "rsc15_train_full.txt"), sep='\t',
                                 dtype={'ItemId': np.int64})
        test_data = pd.read_csv(os.path.join(FLAGS.data_dir, "rsc15_test.txt"), sep='\t', dtype={'ItemId': np.int64})

        itemids = train_data[item_key].unique()
        itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
        test_data.sort([session_key, time_key], inplace=True)
        offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
        offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
        evalutation_point_count = 0
        mrr, recall = 0.0, 0.0
        if len(offset_sessions) - 1 < batch_size:
            batch_size = len(offset_sessions) - 1
        iters = np.arange(batch_size).astype(np.int32)
        maxiter = iters.max()
        start = offset_sessions[iters]
        end = offset_sessions[iters + 1]
        in_idx = np.zeros(batch_size, dtype=np.int32)

        output_file_list = []
        for root, dirs, files in os.walk(output_pre):
            for bin_file in files:
                if str(bin_file).endswith(".bin"):
                    output_file_list.append(os.path.join(root, bin_file))
        output_file_list.sort()
        file_num = len(output_file_list)
        x = 0
        while True:
            valid_mask = iters >= 0
            if valid_mask.sum() == 0:
                break
            start_valid = start[valid_mask]
            minlen = (end[valid_mask] - start_valid).min()
            in_idx[valid_mask] = test_data[item_key].values[start_valid]
            cur_file = x + 1
            if cur_file % 1000 == 0 or cur_file == file_num - 1:
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                "Process predict result %d/%d" % (cur_file, file_num - 1)))
            for i in range(minlen - 1):
                out_idx = test_data[item_key].values[start_valid + i + 1]
                preds = np.fromfile("%s/input_%05d_%05d_output_00_000.bin" % (output_pre, x, i), np.float32).reshape(
                    self.batch, self.n_items).T
                preds = pd.DataFrame(data=preds, index=itemidmap.index)
                preds.fillna(0, inplace=True)
                in_idx[valid_mask] = out_idx
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
                rank_ok = ranks < cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / ranks[rank_ok]).sum()
                evalutation_point_count += len(ranks)
            start = start + minlen - 1
            mask = np.arange(len(iters))[valid_mask & (end - start <= 1)]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(offset_sessions) - 1:
                    iters[idx] = -1
                else:
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[maxiter]
                    end[idx] = offset_sessions[maxiter + 1]
            x += 1
        res = (recall / evalutation_point_count, mrr / evalutation_point_count)
        output_file = os.path.join(output_pre, "%s_precision.txt" % model_name)
        with tf.gfile.GFile(output_file, "w") as writer:
            writer.write('Recall@20: {}\tMRR@20: {}\n'.format(res[0], res[1]))
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                        'Recall@20: {}\tMRR@20: {}\n'.format(res[0], res[1])))
