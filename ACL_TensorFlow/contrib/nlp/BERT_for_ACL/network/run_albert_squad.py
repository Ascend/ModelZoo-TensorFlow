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

"""Run BERT on SQuAD 1.1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import json
import math
import os
import re
import string

import numpy as np
import six
import tensorflow.compat.v1 as tf

from network import fine_tuning_utils, tokenization
from network.albert_en import modeling
from network.classifier_utils import DataProcessor

flags = tf.flags

FLAGS = flags.FLAGS

__all__ = 'SquadProcessor'


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tok_start_to_orig_index,
                 tok_end_to_orig_index,
                 token_is_max_context,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 p_mask=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.p_mask = p_mask
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SquadProcessor(DataProcessor):

    def get_examples(self):
        """Read a SQuAD json file into a list of SquadExample."""
        return self.create_examples(self.read_json(os.path.join(FLAGS.data_dir, "dev-v1.1.json")))

    def get_labels(self):
        return [0, 1]

    @classmethod
    def read_json(cls, input_file):
        with tf.io.gfile.GFile(input_file, "r") as reader:
            lines = json.load(reader)["data"]
        return lines

    @staticmethod
    def create_examples(lines):

        examples = []
        for entry in lines:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    is_impossible = False
                    start_position = None
                    end_position = None
                    orig_answer_text = None

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        paragraph_text=paragraph_text,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples

    def convert_examples_to_features(self, examples, tokenizer, label_list=None, preprocess=False):
        """Loads a data file into a list of `InputBatch`s."""

        def _convert_index(index, pos, m=None, is_start=True):
            """Converts index."""
            if index[pos] is not None:
                return index[pos]
            n = len(index)
            rear = pos
            while rear < n - 1 and index[rear] is None:
                rear += 1
            front = pos
            while front > 0 and index[front] is None:
                front -= 1
            assert index[front] is not None or index[rear] is not None
            if index[front] is None:
                if index[rear] >= 1:
                    if is_start:
                        return 0
                    else:
                        return index[rear] - 1
                return index[rear]
            if index[rear] is None:
                if m is not None and index[front] < m - 1:
                    if is_start:
                        return index[front] + 1
                    else:
                        return m - 1
                return index[front]
            if is_start:
                if index[rear] > index[front] + 1:
                    return index[front] + 1
                else:
                    return index[rear]
            else:
                if index[rear] > index[front] + 1:
                    return index[rear] - 1
                else:
                    return index[front]

        def _check_is_max_context(doc_spans, cur_span_index, position):
            """Check if this is the 'max context' doc span for the token."""

            # Because of the sliding window approach taken to scoring documents, a single
            # token can appear in multiple documents. E.g.
            #  Doc: the man went to the store and bought a gallon of milk
            #  Span A: the man went to the
            #  Span B: to the store and bought
            #  Span C: and bought a gallon of
            #  ...
            #
            # Now the word 'bought' will have two scores from spans B and C. We only
            # want to consider the score with "maximum context", which we define as
            # the *minimum* of its left and right context (the *sum* of left and
            # right context will always be the same, of course).
            #
            # In the example the maximum context for 'bought' would be span C since
            # it has 1 left context and 3 right context, while span B has 4 left context
            # and 0 right context.
            best_score = None
            best_span_index = None
            for (span_index, doc_span) in enumerate(doc_spans):
                end = doc_span.start + doc_span.length - 1
                if position < doc_span.start:
                    continue
                if position > end:
                    continue
                num_left_context = position - doc_span.start
                num_right_context = end - position
                score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
                if best_score is None or score > best_score:
                    best_score = score
                    best_span_index = span_index

            return cur_span_index == best_span_index

        doc_stride = FLAGS.doc_stride
        max_query_length = FLAGS.max_query_length
        do_lower_case = FLAGS.do_lower_case
        max_seq_length = FLAGS.max_seq_length

        cnt_pos, cnt_neg = 0, 0
        unique_id = 1000000000
        max_n, max_m = 1024, 1024
        f = np.zeros((max_n, max_m), dtype=np.float32)
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        unique_ids_list = []
        all_features = []

        for (example_index, example) in enumerate(examples):
            if (example_index + 1) % 1000 == 0 or (example_index + 1) == len(examples):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                                "Reading example from file %s/%s" % (example_index + 1, len(examples))))

            query_tokens = tokenization.encode_ids(
                tokenizer.sp_model,
                tokenization.preprocess_text(example.question_text, lower=do_lower_case))

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            paragraph_text = example.paragraph_text
            para_tokens = tokenization.encode_pieces(
                tokenizer.sp_model,
                tokenization.preprocess_text(example.paragraph_text, lower=do_lower_case),
                return_unicode=False)

            chartok_to_tok_index = []
            tok_start_to_chartok_index = []
            tok_end_to_chartok_index = []
            char_cnt = 0
            para_tokens = [six.ensure_text(token, "utf-8") for token in para_tokens]
            for i, token in enumerate(para_tokens):
                new_token = six.ensure_text(token).replace(
                    tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
                chartok_to_tok_index.extend([i] * len(new_token))
                tok_start_to_chartok_index.append(char_cnt)
                char_cnt += len(new_token)
                tok_end_to_chartok_index.append(char_cnt - 1)

            tok_cat_text = "".join(para_tokens).replace(
                tokenization.SPIECE_UNDERLINE.decode("utf-8"), " ")
            n, m = len(paragraph_text), len(tok_cat_text)

            if n > max_n or m > max_m:
                max_n = max(n, max_n)
                max_m = max(m, max_m)
                f = np.zeros((max_n, max_m), dtype=np.float32)

            g = {}

            def _lcs_match(max_dist, n=n, m=m):
                """Longest-common-substring algorithm."""
                f.fill(0)
                g.clear()

                ### longest common sub sequence
                # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
                for i in range(n):

                    # note(zhiliny):
                    # unlike standard LCS, this is specifically optimized for the setting
                    # because the mismatch between sentence pieces and original text will
                    # be small
                    for j in range(i - max_dist, i + max_dist):
                        if j >= m or j < 0: continue

                        if i > 0:
                            g[(i, j)] = 0
                            f[i, j] = f[i - 1, j]

                        if j > 0 and f[i, j - 1] > f[i, j]:
                            g[(i, j)] = 1
                            f[i, j] = f[i, j - 1]

                        f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                        if (tokenization.preprocess_text(
                                paragraph_text[i], lower=do_lower_case,
                                remove_space=False) == tok_cat_text[j]
                                and f_prev + 1 > f[i, j]):
                            g[(i, j)] = 2
                            f[i, j] = f_prev + 1

            max_dist = abs(n - m) + 5
            for _ in range(2):
                _lcs_match(max_dist)
                if f[n - 1, m - 1] > 0.8 * n: break
                max_dist *= 2

            orig_to_chartok_index = [None] * n
            chartok_to_orig_index = [None] * m
            i, j = n - 1, m - 1
            while i >= 0 and j >= 0:
                if (i, j) not in g: break
                if g[(i, j)] == 2:
                    orig_to_chartok_index[i] = j
                    chartok_to_orig_index[j] = i
                    i, j = i - 1, j - 1
                elif g[(i, j)] == 1:
                    j = j - 1
                else:
                    i = i - 1

            if all(v is None for v in orig_to_chartok_index) or f[n - 1, m - 1] < 0.8 * n:
                continue

            tok_start_to_orig_index = []
            tok_end_to_orig_index = []
            for i in range(len(para_tokens)):
                start_chartok_pos = tok_start_to_chartok_index[i]
                end_chartok_pos = tok_end_to_chartok_index[i]
                start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                                n, is_start=True)
                end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                              n, is_start=False)

                tok_start_to_orig_index.append(start_orig_pos)
                tok_end_to_orig_index.append(end_orig_pos)

            def _piece_to_id(x):
                if six.PY2 and isinstance(x, six.text_type):
                    x = six.ensure_binary(x, "utf-8")
                return tokenizer.sp_model.PieceToId(x)

            all_doc_tokens = list(map(_piece_to_id, para_tokens))

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_is_max_context = {}
                segment_ids = []
                p_mask = []

                cur_tok_start_to_orig_index = []
                cur_tok_end_to_orig_index = []

                tokens.append(tokenizer.sp_model.PieceToId("[CLS]"))
                segment_ids.append(0)
                p_mask.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                    p_mask.append(1)
                tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
                segment_ids.append(0)
                p_mask.append(1)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i

                    cur_tok_start_to_orig_index.append(
                        tok_start_to_orig_index[split_token_index])
                    cur_tok_end_to_orig_index.append(
                        tok_end_to_orig_index[split_token_index])

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                    p_mask.append(0)
                tokens.append(tokenizer.sp_model.PieceToId("[SEP]"))
                segment_ids.append(1)
                p_mask.append(1)

                paragraph_len = len(tokens)
                input_ids = tokens

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    p_mask.append(1)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                span_is_impossible = example.is_impossible
                start_position = None
                end_position = None

                feat_example_index = example_index

                if example_index < 1:
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                    "I", "*** Example ***"))
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                    "I", "unique_id: %s" % unique_id))
                    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                    "I", "input_ids: %s" % " ".join([str(x) for x in input_ids])))

                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)
                unique_ids_list.append([unique_id])

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=feat_example_index,
                    doc_span_index=doc_span_index,
                    tok_start_to_orig_index=cur_tok_start_to_orig_index,
                    tok_end_to_orig_index=cur_tok_end_to_orig_index,
                    token_is_max_context=token_is_max_context,
                    tokens=[tokenizer.sp_model.IdToPiece(x) for x in tokens],
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    p_mask=p_mask)

                all_features.append(feature)

                unique_id += 1
                if span_is_impossible:
                    cnt_neg += 1
                else:
                    cnt_pos += 1

        if preprocess:
            input_ids = []
            input_mask = []
            segment_ids = []
            unique_ids = []

            input_ids_path = os.path.join(FLAGS.output_dir, "input_ids")
            if os.path.exists(input_ids_path):
                os.system("rm -rf %s" % input_ids_path)
            os.makedirs(input_ids_path)
            input_mask_path = os.path.join(FLAGS.output_dir, "input_mask")
            if os.path.exists(input_mask_path):
                os.system("rm -rf %s" % input_mask_path)
            os.makedirs(input_mask_path)
            segment_ids_path = os.path.join(FLAGS.output_dir, "segment_ids")
            if os.path.exists(segment_ids_path):
                os.system("rm -rf %s" % segment_ids_path)
            os.makedirs(segment_ids_path)
            unique_ids_path = os.path.join(FLAGS.output_dir, "unique_ids")
            if os.path.exists(unique_ids_path):
                os.system("rm -rf %s" % unique_ids_path)
            os.makedirs(unique_ids_path)

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
                    unique_ids.extend([unique_ids_list[ex_index]])

                    np.array(input_ids).astype(np.int32).tofile(
                        os.path.join(input_ids_path, 'input_ids_%05d.bin' % ex_index))
                    np.array(input_mask).astype(np.int32).tofile(
                        os.path.join(input_mask_path, 'input_mask_%05d.bin' % ex_index))
                    np.array(segment_ids).astype(np.int32).tofile(
                        os.path.join(segment_ids_path, 'segment_ids_%05d.bin' % ex_index))
                    np.array(unique_ids).astype(np.int32).tofile(
                        os.path.join(unique_ids_path, 'unique_ids_%05d.bin' % ex_index))

                    input_ids = []
                    input_mask = []
                    segment_ids = []
                    unique_ids = []
                else:
                    input_ids.extend(input_ids_list[ex_index])
                    input_mask.extend(input_mask_list[ex_index])
                    segment_ids.extend(segment_ids_list[ex_index])
                    unique_ids.extend([unique_ids_list[ex_index]])

        return all_features

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

        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        batch_size = final_hidden_shape[0]
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        output_weights = tf.get_variable(
            "cls/squad/output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "cls/squad/output_bias", [num_labels], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, num_labels], name="logits")

        return logits

    def calc_precision(self):
        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"])

        # pylint: disable=invalid-name
        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_log_prob", "end_log_prob"])

        def _unstack(a, axis=0):
            return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]

        data_dir = FLAGS.data_dir
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)

        vocab_file = FLAGS.vocab_file
        spm_model_file = FLAGS.spm_model_file
        do_lower_case = FLAGS.do_lower_case
        max_seq_length = FLAGS.max_seq_length
        unique_dir = os.path.join(FLAGS.data_dir, 'unique_ids')
        n_best_size = FLAGS.n_best_size
        max_answer_length = FLAGS.max_answer_length

        examples = self.get_examples()

        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case, spm_model_file)

        features = self.convert_examples_to_features(examples, tokenizer)

        all_results = []
        unique_id_list = []
        start_logits_list = []
        end_logits_list = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_pre = os.path.join(output_dir, model_name)

        unique_file_list = []
        for root, dirs, files in os.walk(unique_dir):
            for bin_file in files:
                if "unique_ids_" in bin_file and bin_file.endswith(".bin"):
                    unique_file_list.append(os.path.join(root, bin_file))
        unique_file_list.sort()

        output_file_list = []
        for root, dirs, files in os.walk(output_pre):
            for bin_file in files:
                if bin_file.endswith(".bin"):
                    output_file_list.append(os.path.join(root, bin_file))
        output_file_list.sort()

        for i in range(len(output_file_list)):
            if (i + 1) % 1000 == 0 or (i + 1) == len(output_file_list):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I",
                                                "Start to load result files: %d/%d" % (i + 1, len(output_file_list))))
            output_pre_idx = output_pre + "_%05d" % i

            unique_id = np.fromfile(unique_file_list[i], np.int32).reshape([1])
            logits = np.fromfile(output_pre_idx + "_output_00_000.bin", np.float32).reshape([1, max_seq_length, 2])
            logits = logits.transpose((2, 0, 1))
            [start_logits, end_logits] = _unstack(logits, axis=0)

            start_logits = [float(x) for x in start_logits.flat]
            end_logits = [float(x) for x in end_logits.flat]

            unique_id_list.extend(unique_id)
            start_logits_list.extend(start_logits)
            end_logits_list.extend(end_logits)

        for idx in range(len(output_file_list)):
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(output_file_list):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I",
                                                "Start to process results: %d/%d" % (idx + 1, len(output_file_list))))
            all_results.append(RawResult(unique_id=int(unique_id_list[idx]),
                                         start_log_prob=start_logits_list[
                                                        idx * max_seq_length: (idx + 1) * max_seq_length],
                                         end_log_prob=end_logits_list[
                                                      idx * max_seq_length: (idx + 1) * max_seq_length]))

        output_prediction_file = os.path.join(output_pre, "%s_precision.txt" % model_name)
        output_nbest_file = os.path.join(output_pre, "%s_nbest_predictions.txt" % model_name)

        num_features = len(features)
        num_results = len(all_results)

        if num_features > num_results:
            features = features[:num_results]
            examples = examples[:num_results]
        else:
            all_results = all_results[:num_features]

        def _get_best_indexes(logits, n_best_size):
            """Get the n-best logits from a list."""
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i in range(len(index_and_score)):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i][0])
            return best_indexes

        def _compute_softmax(scores):
            """Compute softmax probability over raw logits."""
            if not scores:
                return []

            max_score = None
            for score in scores:
                if max_score is None or score > max_score:
                    max_score = score

            exp_scores = []
            total_sum = 0.0
            for score in scores:
                x = math.exp(score - max_score)
                exp_scores.append(x)
                total_sum += x

            probs = []
            for score in exp_scores:
                probs.append(score / total_sum)
            return probs

        def _accumulate_predictions(result_dict, examples, features, all_results, n_best_size, max_answer_length):
            """accumulate predictions for each positions in a dictionary."""
            example_index_to_features = collections.defaultdict(list)
            for feature in features:
                example_index_to_features[feature.example_index].append(feature)

            unique_id_to_result = {}
            for result in all_results:
                unique_id_to_result[result.unique_id] = result

            for (example_index, example) in enumerate(examples):
                if example_index not in result_dict:
                    result_dict[example_index] = {}
                features = example_index_to_features[example_index]

                for (feature_index, feature) in enumerate(features):
                    if feature.unique_id not in result_dict[example_index]:
                        result_dict[example_index][feature.unique_id] = {}
                    result = unique_id_to_result[feature.unique_id]
                    start_indexes = _get_best_indexes(result.start_log_prob, n_best_size)
                    end_indexes = _get_best_indexes(result.end_log_prob, n_best_size)
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            doc_offset = feature.tokens.index("[SEP]") + 1
                            # We could hypothetically create invalid predictions, e.g., predict
                            # that the start of the span is in the question. We throw out all
                            # invalid predictions.
                            if start_index - doc_offset >= len(feature.tok_start_to_orig_index):
                                continue
                            if end_index - doc_offset >= len(feature.tok_end_to_orig_index):
                                continue
                            if not feature.token_is_max_context.get(start_index, False):
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > max_answer_length:
                                continue
                            start_log_prob = result.start_log_prob[start_index]
                            end_log_prob = result.end_log_prob[end_index]
                            start_idx = start_index - doc_offset
                            end_idx = end_index - doc_offset
                            if (start_idx, end_idx) not in result_dict[example_index][feature.unique_id]:
                                result_dict[example_index][feature.unique_id][(start_idx, end_idx)] = []
                            result_dict[example_index][feature.unique_id][(start_idx, end_idx)].append(
                                (start_log_prob, end_log_prob))

        def _write_predictions(result_dict, all_examples, all_features, all_results, n_best_size,
                               output_prediction_file, output_nbest_file):
            """Write final predictions to the json file and log-odds of null if needed."""

            example_index_to_features = collections.defaultdict(list)
            for feature in all_features:
                example_index_to_features[feature.example_index].append(feature)

            unique_id_to_result = {}
            for result in all_results:
                unique_id_to_result[result.unique_id] = result

            all_predictions = collections.OrderedDict()
            all_nbest_json = collections.OrderedDict()

            for (example_index, example) in enumerate(all_examples):
                features = example_index_to_features[example_index]

                prelim_predictions = []
                for (feature_index, feature) in enumerate(features):
                    for ((start_idx, end_idx), logprobs) in \
                            result_dict[example_index][feature.unique_id].items():
                        start_log_prob = 0
                        end_log_prob = 0
                        for logprob in logprobs:
                            start_log_prob += logprob[0]
                            end_log_prob += logprob[1]
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_idx,
                                end_index=end_idx,
                                start_log_prob=start_log_prob / len(logprobs),
                                end_log_prob=end_log_prob / len(logprobs)))

                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_log_prob + x.end_log_prob),
                    reverse=True)

                seen_predictions = {}
                nbest = []
                for pred in prelim_predictions:
                    if len(nbest) >= n_best_size:
                        break
                    feature = features[pred.feature_index]
                    if pred.start_index >= 0:  # this is a non-null prediction
                        tok_start_to_orig_index = feature.tok_start_to_orig_index
                        tok_end_to_orig_index = feature.tok_end_to_orig_index
                        start_orig_pos = tok_start_to_orig_index[pred.start_index]
                        end_orig_pos = tok_end_to_orig_index[pred.end_index]

                        paragraph_text = example.paragraph_text
                        final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()
                        if final_text in seen_predictions:
                            continue

                        seen_predictions[final_text] = True
                    else:
                        final_text = ""
                        seen_predictions[final_text] = True

                    nbest.append(
                        _NbestPrediction(
                            text=final_text,
                            start_log_prob=pred.start_log_prob,
                            end_log_prob=pred.end_log_prob))

                # In very rare edge cases we could have no valid predictions. So we
                # just create a nonce prediction in this case to avoid failure.
                if not nbest:
                    nbest.append(
                        _NbestPrediction(text="empty", start_log_prob=0.0, end_log_prob=0.0))

                assert len(nbest) >= 1

                total_scores = []
                best_non_null_entry = None
                for entry in nbest:
                    total_scores.append(entry.start_log_prob + entry.end_log_prob)
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry

                probs = _compute_softmax(total_scores)

                nbest_json = []
                for (i, entry) in enumerate(nbest):
                    output = collections.OrderedDict()
                    output["text"] = entry.text
                    output["probability"] = probs[i]
                    output["start_log_prob"] = entry.start_log_prob
                    output["end_log_prob"] = entry.end_log_prob
                    nbest_json.append(output)

                assert len(nbest_json) >= 1

                all_predictions[example.qas_id] = nbest_json[0]["text"]
                all_nbest_json[example.qas_id] = nbest_json

            with tf.gfile.GFile(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

            with tf.gfile.GFile(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

            return all_predictions

        result_dict = {}
        _accumulate_predictions(result_dict, examples, features, all_results, n_best_size, max_answer_length)

        _write_predictions(result_dict, examples, features, all_results, n_best_size,
                           output_prediction_file, output_nbest_file)

        def _normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""

            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def _f1_score(prediction, ground_truth):
            prediction_tokens = _normalize_answer(prediction).split()
            ground_truth_tokens = _normalize_answer(ground_truth).split()
            common = (
                    collections.Counter(prediction_tokens)
                    & collections.Counter(ground_truth_tokens))
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def _exact_match_score(prediction, ground_truth):
            return _normalize_answer(prediction) == _normalize_answer(ground_truth)

        def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
            scores_for_ground_truths = []
            for ground_truth in ground_truths:
                score = metric_fn(prediction, ground_truth)
                scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        def _evaluate(dataset, predictions):
            f1 = exact_match = total = 0
            for article in dataset:
                for paragraph in article["paragraphs"]:
                    for qa in paragraph["qas"]:
                        total += 1
                        if qa["id"] not in predictions:
                            continue
                        ground_truths = [x["text"] for x in qa["answers"]]
                        # ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                        prediction = predictions[qa["id"]]
                        exact_match += _metric_max_over_ground_truths(_exact_match_score, prediction, ground_truths)
                        f1 += _metric_max_over_ground_truths(_f1_score, prediction, ground_truths)

            exact_match = 100.0 * exact_match / total
            f1 = 100.0 * f1 / total
            return {"exact_match": exact_match, "f1": f1}

        with open(os.path.join(data_dir, "dev-v1.1.json")) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        with open(output_prediction_file) as prediction_file:
            predictions = json.load(prediction_file)

        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "I", json.dumps(_evaluate(dataset, predictions))))
