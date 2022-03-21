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

import csv
import json
import os

import tensorflow.compat.v1 as tf
from absl import flags

from network import tokenization
from network.classifier_utils import DataProcessor, InputExample

FLAGS = flags.FLAGS

__all__ = 'CmnliProcessor'


class CmnliProcessor(DataProcessor):
    """Processor for the CMNLI data set."""

    def get_examples(self):
        """See base class."""
        return self.create_examples(os.path.join(FLAGS.data_dir, "dev.json"), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def create_examples(file_name, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        lines = tf.gfile.Open(file_name, "r")
        index = 0
        for line in lines:
            line_obj = json.loads(line)
            index = index + 1
            guid = "%s-%s" % (set_type, index)
            text_a = tokenization.convert_to_unicode(line_obj["sentence1"])
            text_b = tokenization.convert_to_unicode(line_obj["sentence2"])
            label = tokenization.convert_to_unicode(line_obj["label"]) if set_type != 'test' else 'neutral'

            if label != "-":
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    @staticmethod
    def read_label(input_file):

        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                data_json = json.loads(line[-1])
                if data_json['label'] == '-':
                    continue
                label_id = label_map[data_json['label']]
                labels.append(label_id)
            return labels
