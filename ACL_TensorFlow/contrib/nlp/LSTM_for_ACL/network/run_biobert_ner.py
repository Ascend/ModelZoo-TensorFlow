#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2019 The BioNLP-HZAU Kaiyin Zhou
# Time:2019/04/08
"""

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

import collections
import datetime
import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags

from network import tokenization, fine_tuning_utils
from network.classifier_utils import DataProcessor, InputExample, InputFeatures

FLAGS = flags.FLAGS

__all__ = 'NerProcessor'


class NerProcessor(DataProcessor):

    def get_examples(self):
        return self.create_example(self.read_tsv(os.path.join(FLAGS.data_dir, "devel.tsv")), "dev")

    def get_labels(self):
        return ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        """Reads a BIO data."""
        inpFilept = open(input_file)
        lines = []
        words = []
        labels = []
        for lineIdx, line in enumerate(inpFilept):
            contents = line.splitlines()[0]
            lineList = contents.split()
            if len(lineList) == 0:  # For blank line
                assert len(words) == len(labels), "lineIdx: %s,  len(words)(%s) != len(labels)(%s) \n %s\n%s" % (
                    lineIdx, len(words), len(labels), " ".join(words), " ".join(labels))
                if len(words) != 0:
                    wordSent = " ".join(words)
                    labelSent = " ".join(labels)
                    lines.append((labelSent, wordSent))
                    words = []
                    labels = []
                else:
                    print("Two continual empty lines detected!")
            else:
                words.append(lineList[0])
                labels.append(lineList[-1])
        if len(words) != 0:
            wordSent = " ".join(words)
            labelSent = " ".join(labels)
            lines.append((labelSent, wordSent))
            words = []
            labels = []

        inpFilept.close()
        return lines

    @staticmethod
    def create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=texts, label=labels))
        return examples

    @staticmethod
    def convert_single_example(example, label_list, tokenizer):
        max_seq_length = FLAGS.max_seq_length
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i
        with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

        textlist = example.text_a.split()
        labellist = example.label.split()
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m, tok in enumerate(token):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")

        # drop if token is longer than max_seq_length
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            # label_mask = label_mask
        )
        return feature

    def convert_examples_to_features(self, examples, tokenizer, label_list=None, preprocess=False):

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        input_label_list = []

        for (example_index, example) in enumerate(examples):
            if (example_index + 1) % 1000 == 0 or (example_index + 1) == len(examples):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                "Reading example from file %s/%s" % (example_index + 1, len(examples))))

            feature = self.convert_single_example(example, label_list, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)

            input_ids = feature.input_ids
            input_mask = feature.input_mask
            segment_ids = feature.segment_ids
            input_label = feature.label_ids

            if example_index < 1:
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "*** Example ***"))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "guid: %s" % example.guid))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "input_ids: %s" % " ".join([str(x) for x in input_ids])))

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            input_label_list.append(input_label)

        if preprocess:
            input_ids = []
            input_mask = []
            segment_ids = []
            input_label = []

            input_ids_path = os.path.join(FLAGS.output_dir, 'input_ids')
            if os.path.exists(input_ids_path):
                os.system("rm -rf %s" % input_ids_path)
            os.makedirs(input_ids_path)

            input_mask_path = os.path.join(FLAGS.output_dir, 'input_mask')
            if os.path.exists(input_mask_path):
                os.system("rm -rf %s" % input_mask_path)
            os.makedirs(input_mask_path)

            segment_ids_path = os.path.join(FLAGS.output_dir, 'segment_ids')
            if os.path.exists(segment_ids_path):
                os.system("rm -rf %s" % segment_ids_path)
            os.makedirs(segment_ids_path)

            label_ids_path = os.path.join(FLAGS.output_dir, 'label_ids')
            if os.path.exists(label_ids_path):
                os.system("rm -rf %s" % label_ids_path)
            os.makedirs(label_ids_path)

            file_num = len(input_ids_list)
            for ex_index in range(len(input_ids_list)):
                cur_file = ex_index + 1
                if cur_file % 1000 == 0 or cur_file == file_num:
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                    "Writing example to file %s/%s" % (int(cur_file), file_num)))

                if (ex_index + 1) % 1 == 0:
                    input_ids.extend(input_ids_list[ex_index])
                    input_mask.extend(input_mask_list[ex_index])
                    segment_ids.extend(segment_ids_list[ex_index])
                    input_label.extend(input_label_list[ex_index])

                    np.array(input_ids).astype(np.int32).tofile(
                        os.path.join(input_ids_path, 'input_ids_%05d.bin' % ex_index))
                    np.array(input_mask).astype(np.int32).tofile(
                        os.path.join(input_mask_path, 'input_mask_%05d.bin' % ex_index))
                    np.array(segment_ids).astype(np.int32).tofile(
                        os.path.join(segment_ids_path, 'segment_ids_%05d.bin' % ex_index))
                    np.array(input_label).astype(np.int32).tofile(
                        os.path.join(label_ids_path, 'label_ids_%05d.bin' % ex_index))

                    input_ids = []
                    input_mask = []
                    segment_ids = []
                    input_label = []
                else:
                    input_ids.extend(input_ids_list[ex_index])
                    input_mask.extend(input_mask_list[ex_index])
                    segment_ids.extend(segment_ids_list[ex_index])
                    input_label.extend(input_label_list[ex_index])

    def create_model(self):
        """Creates a classification model."""
        label_list = self.get_labels()

        input_ids = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'input_mask')
        segment_ids = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'segment_ids')
        num_labels = len(label_list)

        (_, final_hidden) = fine_tuning_utils.create_bert(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        hidden_size = final_hidden.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )

        output_layer = tf.reshape(final_hidden, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        logits = tf.nn.softmax(logits, axis=-1)
        logits = tf.math.argmax(logits, axis=-1, name="logits")

        return logits

    def calc_precision(self):

        data_dir = FLAGS.data_dir
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)
        max_seq_length = FLAGS.max_seq_length

        label_ids = os.path.join(data_dir, 'label_ids')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]

        output_pre = os.path.join(output_dir, model_name)

        label_ids_file_list = []
        for root, dirs, files in os.walk(label_ids):
            for bin_file in files:
                if "label_ids_" in bin_file and bin_file.endswith(".bin"):
                    label_ids_file_list.append(os.path.join(root, bin_file))
        label_ids_file_list.sort()

        output_file_list = []
        for root, dirs, files in os.walk(output_pre):
            for bin_file in files:
                if bin_file.endswith(".bin"):
                    output_file_list.append(os.path.join(root, bin_file))
        output_file_list.sort()

        all_result = []
        all_labels = []
        for idx in range(len(output_file_list)):
            predict = np.fromfile(output_file_list[idx], dtype=np.int64).astype(np.int32).reshape(
                [1, max_seq_length])

            label_ids = np.fromfile(label_ids_file_list[idx], dtype=np.int32).reshape(
                [1, max_seq_length])
            all_result.extend(predict)
            all_labels.extend(label_ids)

        predict_cnt = 0
        acc_cnt = 0
        for idx, label in enumerate(all_labels):
            for i in range(len(label)):
                if label[i] != 0:
                    if label[i] == all_result[idx][i]:
                        acc_cnt += 1
                    predict_cnt += 1

        predict_accuracy = 0.0
        if predict_cnt > 0:
            predict_accuracy = acc_cnt / predict_cnt
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Predict accuracy: %.3f" % predict_accuracy))

        result_save_file = os.path.join(output_pre, "%s_precision.txt" % model_name)
        fp = open(result_save_file, "w")
        fp.write("predict_cnt: %d, correct_cnt: %d\n" % (predict_cnt, acc_cnt))
        fp.write("predict_accuracy: %0.4f\n" % predict_accuracy)
        fp.close()
