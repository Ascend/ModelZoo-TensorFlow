# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-04 14:29:04

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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os

import numpy as np
from absl import flags

from network import tokenization
from network.classifier_utils import DataProcessor, InputExample

FLAGS = flags.FLAGS

__all__ = 'IflytekProcessor'


class IflytekProcessor(DataProcessor):
    """Processor for the iFLYTEK data set."""

    def get_examples(self):
        """See base class."""
        return self.create_examples(self.read_json(os.path.join(FLAGS.data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(119):
            labels.append(str(i))
        return labels

    @staticmethod
    def calc_bert_infer_accuracy(real_labels, output_prefix):
        label_list = real_labels[0]
        real_label_dict = real_labels[1]
        index2label_map = {}
        for (i, label) in enumerate(label_list):
            index2label_map[i] = label
        predict_cnt = 0
        acc_cnt = 0
        output_files = []
        for root, dirs, files in os.walk(output_prefix):
            for file in files:
                if file.endswith('.bin'):
                    output_files.append(os.path.join(root, file))
        output_files.sort()
        for idx, file in enumerate(output_files):
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(output_files):
                print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                                "I", "Start to process results: %d/%d" % (idx + 1, len(output_files))))
            predict_result = np.fromfile(file, dtype=np.float32).tolist()
            label_index = predict_result.index(max(predict_result))
            predict_label = index2label_map[label_index]
            real_label = real_label_dict[str(idx)]
            predict_cnt = predict_cnt + 1
            if predict_label == real_label:
                acc_cnt = acc_cnt + 1

        predict_accuracy = 0.0
        if predict_cnt > 0:
            predict_accuracy = acc_cnt / predict_cnt
            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                            "I", "Predict accuracy: %.3f" % predict_accuracy))

        model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        result_save_file = os.path.join(output_prefix, "%s_precision.txt" % model_name)
        fp = open(result_save_file, "w")
        fp.write("predict_cnt: %d, correct_cnt: %d\n" % (predict_cnt, acc_cnt))
        fp.write("predict_accuracy: %0.4f\n" % predict_accuracy)
        fp.close()

    @staticmethod
    def create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line['sentence'])
            text_b = None
            label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else "0"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def calc_precision(self):

        def _read_to_json(input_file):
            with open(input_file, "r") as input_f:
                reader = input_f.readlines()
                lines = []
                for line in reader:
                    lines.append(json.loads(line.strip()))
                return lines

        def _get_label_list(label_list_file):
            labels_lines = _read_to_json(label_list_file)
            label_list = []
            for i, line in enumerate(labels_lines):
                label_id = str(line['label'])
                label_list.append(label_id)
            return label_list

        def _get_real_label(input_file):
            examples_lines = _read_to_json(input_file)
            real_label_dict = {}
            for i, line in enumerate(examples_lines):
                label_id = str(line['label'])
                real_label_dict[str(i)] = label_id
            return real_label_dict

        reference_file = os.path.join(FLAGS.data_dir, "dev.json")
        labels_file = os.path.join(FLAGS.data_dir, "labels.json")

        output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name)
        try:
            model_name = (FLAGS.om_model_file.split('/')[-1]).split('.')[0]
        except AttributeError:
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
        output_prefix = os.path.join(output_dir, model_name)

        real_label = [_get_label_list(labels_file), _get_real_label(reference_file)]

        self.calc_bert_infer_accuracy(real_label, output_prefix)
