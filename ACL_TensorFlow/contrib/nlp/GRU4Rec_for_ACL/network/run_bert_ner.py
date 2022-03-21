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
import re
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from tensorflow.contrib import crf as contrib_crf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import rnn as contrib_rnn

from network import tokenization, fine_tuning_utils
from network.classifier_utils import DataProcessor, InputExample, InputFeatures

FLAGS = flags.FLAGS

__all__ = 'NerProcessor'

ANY_SPACE = '<SPACE>'


class BlstmCrf(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers,
                 num_labels, seq_length, lengths):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param lengths: [batch_size] 每个batch下序列的真实长度
        """
        self.hidden_unit = hidden_unit
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value

    def add_blstm_crf_layer(self):
        """
        blstm-crf网络
        :return:
        """
        # blstm
        lstm_output = self.blstm_layer(self.embedded_chars)
        # project
        logits = self.project_bilstm_layer(lstm_output)
        # crf
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transition",
                shape=[self.num_labels, self.num_labels],
                initializer=contrib_layers.xavier_initializer()
            )
        # CRF decode, pred_ids 是一条最大概率的标注路径
        logits, _ = contrib_crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return logits

    def blstm_layer(self, embedding_chars):
        """
        :return:
        """
        embedding_chars = tf.transpose(embedding_chars, [1, 0, 2])
        max_seq_len = embedding_chars.shape[0].value
        batch = embedding_chars.shape[1].value
        sequence_length = tf.constant([max_seq_len] * batch)
        with tf.variable_scope('rnn_layer'):
            fwd_cell = contrib_rnn.LSTMBlockFusedCell(self.hidden_unit, dtype=tf.float32)
            cell_fw, _ = fwd_cell(embedding_chars, dtype=tf.float32, sequence_length=sequence_length)
            embedding_chars_r = tf.reverse_sequence(embedding_chars, sequence_length, batch_axis=1, seq_axis=0)
            bak_cell = contrib_rnn.LSTMBlockFusedCell(self.hidden_unit, dtype=tf.float32)
            cell_bw_r, _ = bak_cell(embedding_chars_r, dtype=tf.float32, sequence_length=sequence_length)
            cell_bw = tf.reverse_sequence(cell_bw_r, sequence_length, batch_axis=1, seq_axis=0)
            outputs = tf.concat([cell_fw, cell_bw], axis=-1)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=contrib_layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=contrib_layers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])


class FormatError(Exception):
    pass


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_tags = 0  # number of correct chunk tags
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


class NerProcessor(DataProcessor):

    def get_examples(self):
        return self.create_example(self.read_txt(os.path.join(FLAGS.data_dir, "test.txt")), "test")

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
            "[PAD]" for padding
        :return:
        """
        return ["[PAD]", "B-MISC", "I-MISC", "O", "B-PER", "I-PER",
                "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

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
        """
        :param example:
        :param label_list: all labels
        :param max_seq_length:
        :param tokenizer: WordPiece tokenization
        :return: feature

        IN this part we should rebuild input sentences to the following format.
        example:[Jim,Hen,##son,was,a,puppet,##eer]
        labels: [I-PER,I-PER,X,O,O,O,X]

        """
        max_seq_length = FLAGS.max_seq_length
        label_map = {}
        # here start with zero this means that "[PAD]" is zero
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        with open(os.path.join(FLAGS.data_dir, "label2id.pkl"), 'wb') as w:
            pickle.dump(label_map, w)
        text_list = example.text_a.split(' ')
        label_list = example.label.split(' ')
        tokens = []
        labels = []
        for i, (word, label) in enumerate(zip(text_list, label_list)):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for j, _ in enumerate(token):
                if j == 0:
                    labels.append(label)
                else:
                    labels.append("X")
        # only Account for [CLS] with "- 1".
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 1)]
            labels = labels[0:(max_seq_length - 1)]
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
        # after that we don't add "[SEP]" because we want a sentence don't have
        # stop tag, because i think its not very necessary.
        # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # use zero to padding and you should
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
        assert len(ntokens) == max_seq_length
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
        )
        # we need ntokens because if we do output it can help us return to original token.
        return feature, ntokens, label_ids

    def convert_examples_to_features(self, examples, tokenizer, label_list=None, preprocess=False):

        batch_tokens = []
        batch_labels = []

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        input_label_list = []

        for (example_index, example) in enumerate(examples):
            if (example_index + 1) % 1000 == 0 or (example_index + 1) == len(examples):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                "Reading example from file %s/%s" % (example_index + 1, len(examples))))

            feature, ntokens, label_ids = self.convert_single_example(example, label_list, tokenizer)
            batch_tokens.extend(ntokens)
            batch_labels.extend(label_ids)

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

        if FLAGS.classifier_type.lower() == 'crf':
            linear = tf.keras.layers.Dense(num_labels, activation=None)
            logits = linear(final_hidden)
            mask2len = tf.reduce_sum(input_mask, axis=1)

            with tf.variable_scope("crf_loss"):
                trans = tf.get_variable(
                    "transition",
                    shape=[num_labels, num_labels],
                    initializer=contrib_layers.xavier_initializer()
                )

            logits, viterbi_score = contrib_crf.crf_decode(logits, trans, mask2len)
        elif FLAGS.classifier_type.lower() == 'bilstm_crf':
            max_seq_lenght = final_hidden.shape[1].value
            used = tf.sign(tf.abs(input_ids))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            blstm_crf = BlstmCrf(embedded_chars=final_hidden, hidden_unit=1, cell_type='lstm',
                                 num_layers=1, num_labels=num_labels, seq_lenght=max_seq_lenght, lenghts=lengths)
            logits = blstm_crf.add_blstm_crf_layer()
        else:
            linear = tf.keras.layers.Dense(num_labels, activation=None)
            logits = linear(final_hidden)
            logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
            logits = tf.reshape(logits, [-1, num_labels])

            # output not mask we could filtered it in the prediction part.
            logits = tf.math.softmax(logits, axis=-1)
            logits = tf.math.argmax(logits, axis=-1, name="logits")
        return logits

    def calc_precision(self):

        def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, idx):
            token = batch_tokens[idx]
            predict = id2label[prediction]
            true_l = id2label[batch_labels[idx]]
            if token != "[PAD]" and token != "[CLS]" and true_l != "X":
                if predict == "X" and not predict.startswith("##"):
                    predict = "O"
                line = "{}\t{}\t{}\n".format(token, true_l, predict)
                wf.write(line)

        data_dir = FLAGS.data_dir
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)
        vocab_file = FLAGS.vocab_file
        do_lower_case = FLAGS.do_lower_case
        max_seq_length = FLAGS.max_seq_length

        input_ids = os.path.join(data_dir, 'input_ids')
        label_ids = os.path.join(data_dir, 'label_ids')
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_pre = os.path.join(output_dir, model_name)
        output_predict_txt = os.path.join(output_pre, "%s_result.txt" % model_name)
        if os.path.exists(output_predict_txt):
            os.remove(output_predict_txt)

        label_ids_file_list = []
        for root, dirs, files in os.walk(label_ids):
            for bin_file in files:
                if "label_ids_" in bin_file and bin_file.endswith(".bin"):
                    label_ids_file_list.append(os.path.join(root, bin_file))
        label_ids_file_list.sort()

        input_ids_file_list = []
        for root, dirs, files in os.walk(input_ids):
            for bin_file in files:
                if "input_ids_" in bin_file and bin_file.endswith(".bin"):
                    input_ids_file_list.append(os.path.join(root, bin_file))
        input_ids_file_list.sort()

        output_file_list = []
        for root, dirs, files in os.walk(output_pre):
            for bin_file in files:
                if bin_file.endswith(".bin"):
                    output_file_list.append(os.path.join(root, bin_file))
        output_file_list.sort()

        for idx in range(len(output_file_list)):
            predict = np.fromfile(output_file_list[idx], dtype=np.int64).astype(np.int32).reshape([1, max_seq_length])

            input_ids = np.fromfile(input_ids_file_list[idx], dtype=np.int32).reshape([1, max_seq_length])

            label_ids = np.fromfile(label_ids_file_list[idx], dtype=np.int32).reshape([1, max_seq_length])

            batch_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            with open(os.path.join(data_dir, "label2id.pkl"), 'rb') as rf:
                label2id = pickle.load(rf)
                id2label = {value: key for key, value in label2id.items()}

            with open(output_predict_txt, 'a+') as wf:
                for i, prediction in enumerate(predict[0]):
                    _write_base(batch_tokens, id2label, prediction, label_ids[0], wf, i)

        Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

        def _parse_args(argv):
            import argparse
            parser = argparse.ArgumentParser(description='evaluate tagging results using CoNLL criteria',
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            arg = parser.add_argument
            arg('-b', '--boundary', metavar='STR', default='-X-',
                help='sentence boundary')
            arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
                help='character delimiting items in input')
            arg('-o', '--otag', metavar='CHAR', default='O',
                help='alternative outside tag')
            arg('file', nargs='?', default=None)
            return parser.parse_args(argv)

        def _parse_tag(t):
            m = re.match(r'^([^-]*)-(.*)$', t)
            return m.groups() if m else (t, '')

        def _evaluate(iterable):

            options = _parse_args([])  # use defaults

            counts = EvalCounts()
            num_features = None  # number of features per line
            in_correct = False  # currently processed chunks is correct until now
            last_correct = 'O'  # previous chunk tag in corpus
            last_correct_type = ''  # type of previously identified chunk tag
            last_guessed = 'O'  # previously identified chunk tag
            last_guessed_type = ''  # type of previous chunk tag in corpus

            for line in iterable:
                line = line.rstrip('\r\n')

                if options.delimiter == ANY_SPACE:
                    features = line.split()
                else:
                    features = line.split(options.delimiter)

                if num_features is None:
                    num_features = len(features)
                elif num_features != len(features) and len(features) != 0:
                    raise FormatError('unexpected number of features: %d (%d)' % (len(features), num_features))

                if len(features) == 0 or features[0] == options.boundary:
                    features = [options.boundary, 'O', 'O']
                if len(features) < 3:
                    raise FormatError('unexpected number of features in line %s' % line)

                guessed, guessed_type = _parse_tag(features.pop())
                correct, correct_type = _parse_tag(features.pop())
                first_item = features.pop(0)

                if first_item == options.boundary:
                    guessed = 'O'

                end_correct = _end_of_chunk(last_correct, correct,
                                            last_correct_type, correct_type)
                end_guessed = _end_of_chunk(last_guessed, guessed,
                                            last_guessed_type, guessed_type)
                start_correct = _start_of_chunk(last_correct, correct,
                                                last_correct_type, correct_type)
                start_guessed = _start_of_chunk(last_guessed, guessed,
                                                last_guessed_type, guessed_type)

                if in_correct:
                    if (end_correct and end_guessed and
                            last_guessed_type == last_correct_type):
                        in_correct = False
                        counts.correct_chunk += 1
                        counts.t_correct_chunk[last_correct_type] += 1
                    elif end_correct != end_guessed or guessed_type != correct_type:
                        in_correct = False

                if start_correct and start_guessed and guessed_type == correct_type:
                    in_correct = True

                if start_correct:
                    counts.found_correct += 1
                    counts.t_found_correct[correct_type] += 1
                if start_guessed:
                    counts.found_guessed += 1
                    counts.t_found_guessed[guessed_type] += 1
                if first_item != options.boundary:
                    if correct == guessed and guessed_type == correct_type:
                        counts.correct_tags += 1
                    counts.token_counter += 1

                last_guessed = guessed
                last_correct = correct
                last_guessed_type = guessed_type
                last_correct_type = correct_type

            if in_correct:
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1

            return counts

        def _uniq(iterable):
            seen = set()
            return [i for i in iterable if not (i in seen or seen.add(i))]

        def _calculate_metrics(correct, guessed, total):
            tp, fp, fn = correct, guessed - correct, total - correct
            p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
            r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
            f = 0 if p + r == 0 else 2 * p * r / (p + r)
            return Metrics(tp, fp, fn, p, r, f)

        def _metrics(counts):
            c = counts
            overall = _calculate_metrics(
                c.correct_chunk, c.found_guessed, c.found_correct
            )
            by_type = {}
            for t in _uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
                by_type[t] = _calculate_metrics(
                    c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
                )
            return overall, by_type

        def _report(counts, out=None):
            output_file = os.path.join(output_pre, "%s_precision.txt" % model_name)

            if out is None:
                out = output_file
            with open(out, 'w') as out_f:
                overall, by_type = _metrics(counts)

            c = counts
            out_f.write('processed %d tokens with %d phrases; ' %
                      (c.token_counter, c.found_correct))
            out_f.write('found: %d phrases; correct: %d.\n' %
                      (c.found_guessed, c.correct_chunk))

            if c.token_counter > 0:
                out_f.write('accuracy: %6.2f%%; ' %
                          (100. * c.correct_tags / c.token_counter))
                out_f.write('precision: %6.2f%%; ' % (100. * overall.prec))
                out_f.write('recall: %6.2f%%; ' % (100. * overall.rec))
                out_f.write('FB1: %6.2f\n' % (100. * overall.fscore))

            for i, m in sorted(by_type.items()):
                out_f.write('%17s: ' % i)
                out_f.write('precision: %6.2f%%; ' % (100. * m.prec))
                out_f.write('recall: %6.2f%%; ' % (100. * m.rec))
                out_f.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_found_guessed[i]))

        def _end_of_chunk(prev_tag, tag, prev_type, type_):
            # check if a chunk ended between the previous and current word
            # arguments: previous and current chunk tags, previous and current types
            chunk_end = False

            if prev_tag == 'E':
                chunk_end = True
            if prev_tag == 'S':
                chunk_end = True

            if prev_tag == 'B' and tag == 'B':
                chunk_end = True
            if prev_tag == 'B' and tag == 'S':
                chunk_end = True
            if prev_tag == 'B' and tag == 'O':
                chunk_end = True
            if prev_tag == 'I' and tag == 'B':
                chunk_end = True
            if prev_tag == 'I' and tag == 'S':
                chunk_end = True
            if prev_tag == 'I' and tag == 'O':
                chunk_end = True

            if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
                chunk_end = True

            # these chunks are assumed to have length 1
            if prev_tag == ']':
                chunk_end = True
            if prev_tag == '[':
                chunk_end = True

            return chunk_end

        def _start_of_chunk(prev_tag, tag, prev_type, type_):
            # check if a chunk started between the previous and current word
            # arguments: previous and current chunk tags, previous and current types
            chunk_start = False

            if tag == 'B':
                chunk_start = True
            if tag == 'S':
                chunk_start = True

            if prev_tag == 'E' and tag == 'E':
                chunk_start = True
            if prev_tag == 'E' and tag == 'I':
                chunk_start = True
            if prev_tag == 'S' and tag == 'E':
                chunk_start = True
            if prev_tag == 'S' and tag == 'I':
                chunk_start = True
            if prev_tag == 'O' and tag == 'E':
                chunk_start = True
            if prev_tag == 'O' and tag == 'I':
                chunk_start = True

            if tag != 'O' and tag != '.' and prev_type != type_:
                chunk_start = True

            # these chunks are assumed to have length 1
            if tag == '[':
                chunk_start = True
            if tag == ']':
                chunk_start = True

            return chunk_start

        with open(output_predict_txt) as f:
            counts = _evaluate(f)
        _report(counts)
