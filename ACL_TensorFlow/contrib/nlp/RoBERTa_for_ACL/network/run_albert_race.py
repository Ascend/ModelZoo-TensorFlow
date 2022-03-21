# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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

"""ALBERT finetuning runner with sentence piece tokenization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import json
import os

import tensorflow.compat.v1 as tf

from network import fine_tuning_utils
from network.classifier_utils import DataProcessor, PaddingInputExample, InputFeatures

flags = tf.flags

FLAGS = flags.FLAGS

__all__ = 'RaceProcessor'


class RaceExample(object):
    """A single training/test example for the RACE dataset."""

    def __init__(self,
                 example_id,
                 context_sentence,
                 start_ending,
                 endings,
                 label=None):
        self.example_id = example_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = endings
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.example_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.create_examples()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["A", "B", "C", "D"]

    @classmethod
    def read_list(cls, level):
        cur_dir = os.path.join(FLAGS.data_dir, level)
        cur_path_list = []
        all_txt_path = os.path.join(cur_dir, "all.txt")

        if not os.path.exists(all_txt_path):
            file_list = os.listdir(cur_dir)
            for file_name in file_list:
                if not file_name.endswith('.txt'):
                    continue
                cur_path_list.append(os.path.join(cur_dir, file_name))
        else:
            cur_path_list = [all_txt_path]

        cur_path_list.sort()
        return cur_path_list

    def create_examples(self):
        """Read examples from RACE json files."""
        examples = []
        for level in ["middle", "high"]:
            cur_path_list = self.read_list(level)
            for cur_path in cur_path_list:

                with tf.gfile.Open(cur_path) as f:
                    for line in f:
                        cur_data = json.loads(line.strip())

                        answers = cur_data["answers"]
                        options = cur_data["options"]
                        questions = cur_data["questions"]
                        context = self.process_text(cur_data["article"])

                        for i in range(len(answers)):
                            label = ord(answers[i]) - ord("A")
                            qa_list = []

                            question = self.process_text(questions[i])
                            for j in range(4):
                                option = self.process_text(options[i][j])

                                if "_" in question:
                                    qa_cat = question.replace("_", option)
                                else:
                                    qa_cat = " ".join([question, option])

                                qa_list.append(qa_cat)

                            examples.append(
                                RaceExample(
                                    example_id=cur_data["id"],
                                    context_sentence=context,
                                    start_ending=None,
                                    endings=[qa_list[0], qa_list[1], qa_list[2], qa_list[3]],
                                    label=label
                                )
                            )

        return examples

    @staticmethod
    def convert_single_example(example, label_list, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        # RACE is a multiple choice task. To perform this task using AlBERT,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given RACE example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        label_size = len(label_list)
        max_seq_length = FLAGS.max_seq_length
        max_query_length = FLAGS.max_query_length

        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                example_ids=0,
                input_ids=[[0] * max_seq_length] * label_size,
                input_mask=[[0] * max_seq_length] * label_size,
                segment_ids=[[0] * max_seq_length] * label_size,
                label_ids=0,
                is_real_example=False)
        else:
            context_tokens = tokenizer.tokenize(example.context_sentence)
            if example.start_ending is not None:
                start_ending_tokens = tokenizer.tokenize(example.start_ending)

            all_input_tokens = []
            all_input_ids = []
            all_input_mask = []
            all_segment_ids = []
            for ending in example.endings:
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                if example.start_ending is not None:
                    ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
                else:
                    ending_tokens = tokenizer.tokenize(ending)
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                ending_tokens = ending_tokens[- max_query_length:]

                if len(context_tokens_choice) + len(ending_tokens) > max_seq_length - 3:
                    context_tokens_choice = context_tokens_choice[: (
                            max_seq_length - 3 - len(ending_tokens))]
                tokens = ["[CLS]"] + context_tokens_choice + (
                        ["[SEP]"] + ending_tokens + ["[SEP]"])
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (
                        len(ending_tokens) + 1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                all_input_tokens.append(tokens)
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)

            label = example.label

            return InputFeatures(
                example_ids=example.example_id,
                input_ids=all_input_ids,
                input_mask=all_input_mask,
                segment_ids=all_segment_ids,
                label_ids=label
            )

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
            features["input_ids"] = create_int_feature(sum(feature.input_ids, []))
            features["input_mask"] = create_int_feature(sum(feature.input_mask, []))
            features["segment_ids"] = create_int_feature(sum(feature.segment_ids, []))
            features["label_ids"] = create_int_feature([feature.label_ids])
            features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

            input_ids = sum(feature.input_ids, [])
            input_mask = sum(feature.input_mask, [])
            segment_ids = sum(feature.segment_ids, [])
            input_label = [feature.label_ids]

            if example_index < 1:
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "*** Example ***"))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "example_id: %s" % feature.example_ids))
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "input_ids: %s" % " ".join([str(x) for x in input_ids])))

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            input_label_list.append(input_label)

        if preprocess:
            self.dump_to_bin(input_ids_list, input_mask_list, segment_ids_list)

    def create_model(self):
        """Creates a classification model."""
        label_list = self.get_labels()
        num_labels = len(label_list)

        input_ids = tf.placeholder(tf.int32, (None, num_labels * FLAGS.max_seq_length), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, num_labels * FLAGS.max_seq_length), 'input_mask')
        segment_ids = tf.placeholder(tf.int32, (None, num_labels * FLAGS.max_seq_length), 'segment_ids')

        bsz_per_core = tf.shape(input_ids)[0]

        input_ids = tf.reshape(input_ids, [bsz_per_core * num_labels, FLAGS.max_seq_length])
        input_mask = tf.reshape(input_mask, [bsz_per_core * num_labels, FLAGS.max_seq_length])
        token_type_ids = tf.reshape(segment_ids, [bsz_per_core * num_labels, FLAGS.max_seq_length])

        (output_layer, _) = fine_tuning_utils.create_bert(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=token_type_ids)

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [1],
            initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [bsz_per_core, num_labels])
        logits = tf.nn.softmax(logits, axis=-1, name="logits")

        return logits

    def read_label(self, input_file):
        labels = []
        for level in ["middle", "high"]:
            cur_path_list = self.read_list(level)
            for cur_path in cur_path_list:
                with tf.gfile.Open(cur_path) as f:
                    for line in f:
                        cur_data = json.loads(line.strip())
                        answers = cur_data["answers"]
                        for i in range(len(answers)):
                            labels.append(ord(answers[i]) - ord("A"))
        return labels
