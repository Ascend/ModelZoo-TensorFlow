# coding=utf-8
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from tensorflow.contrib import rnn as contrib_rnn

FLAGS = flags.FLAGS

__all__ = 'ImdbProcessor'


class ImdbProcessor(object):
    def __init__(self):

        self.data_dir = FLAGS.data_dir
        self.output_dir = FLAGS.output_dir
        self.predict_batch_size = FLAGS.predict_batch_size
        self.max_seq_len = 250
        self.lstm_units = 64
        self.num_classes = 2
        self.num_dimensions = 50
        self.ids = np.load(os.path.join(self.data_dir, 'idsMatrix.npy'))
        self.word_vectors = np.load(os.path.join(self.data_dir, 'wordVectors.npy'))
        self.is_finetune = True
        self.is_training = False

    def get_features(self, preprocess):
        input_ids_list = []
        label_ids_list = []

        features = dict()

        # 原脚本用1~11499，13500~25000作为训练集，11500~13499作为预测集
        for index in range(1, len(self.ids)):
            if index not in range(11500, 13500):
                continue
            if index in range(11500, 12500):
                label_ids_list.append([1, 0])
            else:
                label_ids_list.append([0, 1])
            input_ids_list.append(self.ids[index - 1:index])

        features['input_ids'] = input_ids_list
        features['label_ids'] = label_ids_list

        if preprocess:
            input_ids = []
            label_ids = []

            input_ids_path = os.path.join(self.output_dir, 'input_ids')
            if os.path.exists(input_ids_path):
                os.system("rm -rf %s" % input_ids_path)
            os.makedirs(input_ids_path)

            label_ids_path = os.path.join(self.output_dir, 'label_ids')
            if os.path.exists(label_ids_path):
                os.system("rm -rf %s" % label_ids_path)
            os.makedirs(label_ids_path)

            file_num = len(input_ids_list)
            for ex_index in range(len(input_ids_list)):
                cur_file = (ex_index + 1)
                if cur_file % 1000 == 0 or cur_file == file_num:
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                    "Writing example to file %s/%s" % (int(cur_file), file_num)))

                if (ex_index + 1) % 1 == 0:
                    input_ids.append(input_ids_list[ex_index])
                    label_ids.append(label_ids_list[ex_index])
                    np.array(input_ids).astype(np.int32).tofile(
                        os.path.join(input_ids_path, 'input_ids_%05d.bin' % ex_index))
                    np.array(label_ids).astype(np.int32).tofile(
                        os.path.join(label_ids_path, 'label_ids_%05d.bin' % ex_index))

                    input_ids = []
                    label_ids = []
                else:
                    input_ids.append(input_ids_list[ex_index])
                    label_ids.append(label_ids_list[ex_index])

        return features

    def create_model(self):
        input_ids = tf.placeholder(tf.int32, [self.predict_batch_size, self.max_seq_len], name='input_ids')

        data = tf.Variable(tf.zeros([self.predict_batch_size, self.max_seq_len, self.num_dimensions]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.word_vectors, input_ids)

        lstm_cell = contrib_rnn.BasicLSTMCell(self.lstm_units, state_is_tuple=True)

        if self.is_training:
            lstm_cell = contrib_rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)

        # https://github.com/HqWei/Sentiment-Analysis 模型中给了两个训练脚本，LSTM/train.py 和 LSTM/train_test.py
        # 由于train_test.py需要加载train.py的ckpt进行训练，此处暂定train.py为预训练脚本，train_test.py为fine tune脚本
        if self.is_finetune:
            # fine tune脚本中多定义了一个initial_state变量
            initial_state = lstm_cell.zero_state(self.predict_batch_size, tf.float32)
            value, _ = tf.nn.dynamic_rnn(lstm_cell, data, initial_state=initial_state, dtype=tf.float32)
        else:
            value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstm_units, self.num_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        logits = (tf.matmul(last, weight) + bias)

        logits = tf.argmax(logits, 1, output_type=tf.dtypes.int32, name='logits')

        return logits

    def calc_precision(self):
        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]

        output_prefix = os.path.join(self.output_dir, FLAGS.task_name, model_name)
        output_files = []
        for root, dirs, files in os.walk(output_prefix):
            for file in files:
                if file.endswith('.bin'):
                    output_files.append(os.path.join(root, file))
        output_files.sort()

        features = self.get_features(False)
        labels_ids = features['label_ids']
        real_labels = []
        for s in range(len(labels_ids)):
            real_labels.append(np.argmax(labels_ids[s]))

        predict_result = []
        for idx, file in enumerate(output_files):
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(output_files):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "Start to process results: %d/%d" % (idx + 1, len(output_files))))
            predict_result.extend(np.fromfile(file, dtype=np.int32).tolist())

        predict_cnt = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for idx, value in enumerate(predict_result):
            if real_labels[idx] == 1:
                if value == real_labels[idx]:
                    tp += 1
                else:
                    fp += 1
            else:
                if value == real_labels[idx]:
                    tn += 1
                else:
                    fn += 1
            predict_cnt += 1

        acc_cnt = tp + tn
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if predict_cnt > 0:
            accuracy = acc_cnt / predict_cnt
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Predict samples: %d" % predict_cnt))
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "accuracy: %.3f; precision: %.3f; recall: %.3f; f1: %.3f" %
                                            (accuracy, precision, recall, f1)))

        model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)

        result_save_file = os.path.join(output_dir, model_name, "%s_precision.txt" % model_name)
        fp = open(result_save_file, "w")
        fp.write("Predict samples: %d, correct samples: %d\n" % (predict_cnt, acc_cnt))
        fp.write("Predict accuracy: %.3f; precision: %.3f; recall: %.3f; f1: %.3f\n" %
                 (accuracy, precision, recall, f1))
        fp.close()
