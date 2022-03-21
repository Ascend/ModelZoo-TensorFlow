# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
# Lint as: python3

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

"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import datetime
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags

from network import fine_tuning_utils, tokenization

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: string. The untokenized text of the second sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/output on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 guid=None,
                 example_ids=None,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.guid = guid
        self.example_ids = example_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self):
        super(DataProcessor, self).__init__()
        self.use_spm = True if FLAGS.spm_model_file is not None else False

    def get_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def process_text(self, text):
        if self.use_spm:
            return tokenization.preprocess_text(text, lower=FLAGS.do_lower_case)
        else:
            return tokenization.convert_to_unicode(text)

    @staticmethod
    def dump_to_bin(input_ids_list, input_mask_list, segment_ids_list):

        input_ids = []
        input_mask = []
        segment_ids = []

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

        file_num = len(input_ids_list)
        for ex_index in range(len(input_ids_list)):
            cur_file = (ex_index + 1)
            if cur_file % 1000 == 0 or cur_file == file_num:
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                "Writing example to file %s/%s" % (int(cur_file), file_num)))

            if (ex_index + 1) % 1 == 0:
                input_ids.extend(input_ids_list[ex_index])
                input_mask.extend(input_mask_list[ex_index])
                segment_ids.extend(segment_ids_list[ex_index])

                np.array(input_ids).astype(np.int32).tofile(
                    os.path.join(input_ids_path, 'input_ids_%05d.bin' % ex_index))
                np.array(input_mask).astype(np.int32).tofile(
                    os.path.join(input_mask_path, 'input_mask_%05d.bin' % ex_index))
                np.array(segment_ids).astype(np.int32).tofile(
                    os.path.join(segment_ids_path, 'segment_ids_%05d.bin' % ex_index))

                input_ids = []
                input_mask = []
                segment_ids = []
            else:
                input_ids.extend(input_ids_list[ex_index])
                input_mask.extend(input_mask_list[ex_index])
                segment_ids.extend(segment_ids_list[ex_index])

    def convert_examples_to_features(self, examples, tokenizer, label_list=None, preprocess=False):

        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []

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
            features["label_ids"] = create_int_feature([feature.label_ids])
            features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

            if example_index < 1:
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "*** Example ***"))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "guid: %s" % example.guid))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "input_ids: %s" % " ".join([str(x) for x in feature.input_ids])))

            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)
            label_ids.append([feature.label_ids])

        if preprocess:
            self.dump_to_bin(input_ids, input_mask, segment_ids)

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def read_json(cls, input_file):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def read_txt(cls, input_file):
        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = []
        words = []
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip()) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l, w))
                words = []
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines

    def create_model(self):
        """Creates a classification model."""

        label_list = self.get_labels()

        input_ids = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'input_mask')
        segment_ids = tf.placeholder(tf.int32, (None, FLAGS.max_seq_length), 'segment_ids')
        num_labels = len(label_list)

        (output_layer, _) = fine_tuning_utils.create_bert(input_ids, input_mask, segment_ids)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.nn.softmax(logits, axis=-1, name='logits')

        return logits

    @staticmethod
    def convert_single_example(example, label_list, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        max_seq_length = FLAGS.max_seq_length

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_ids=0,
                is_real_example=False)
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_ids = label_map[example.label]

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            is_real_example=True)
        return feature

    @staticmethod
    def calc_bert_infer_accuracy(real_labels, output_prefix):
        predict_cnt = 0
        acc_cnt = 0
        output_files = []
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name, output_prefix)
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".bin"):
                    output_files.append(os.path.join(root, file))
        output_files.sort()

        for idx, file in enumerate(output_files):
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(output_files):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "Start to process results: %d/%d" % (idx + 1, len(output_files))))
            predict_result = np.fromfile(file, dtype=np.float32).tolist()
            label_index = predict_result.index(max(predict_result))
            if label_index == real_labels[predict_cnt]:
                acc_cnt += 1
            predict_cnt += 1

        predict_accuracy = 0.0
        if predict_cnt > 0:
            predict_accuracy = acc_cnt / predict_cnt
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Predict accuracy: %.3f" % predict_accuracy))

        model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        result_save_file = os.path.join(output_dir, "%s_precision.txt" % model_name)
        fp = open(result_save_file, "w")
        fp.write("predict_cnt: %d, correct_cnt: %d\n" % (predict_cnt, acc_cnt))
        fp.write("predict_accuracy: %0.4f\n" % predict_accuracy)
        fp.close()

    @staticmethod
    def read_label(input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                labels.append(int(line[1]))
            return labels

    def calc_precision(self):
        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_prefix = model_name
        file_name = "dev.tsv"

        for root, dirs, files in os.walk(FLAGS.data_dir):
            for file in files:
                if "dev" in file or "test" in file:
                    file_name = file

        reference_file = os.path.join(FLAGS.data_dir, file_name)
        real_labels = self.read_label(reference_file)

        self.calc_bert_infer_accuracy(real_labels, output_prefix)
