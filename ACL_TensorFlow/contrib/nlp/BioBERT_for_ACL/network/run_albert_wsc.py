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

__all__ = 'WscProcessor'


class WscProcessor(DataProcessor):
    """Processor for the WSC data set."""

    def get_examples(self):
        """See base class."""
        return self.create_examples(self.read_json(os.path.join(FLAGS.data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    @staticmethod
    def create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line['text'])
            text_a_list = list(text_a)
            target = line['target']
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']

            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)

            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

            text_a = "".join(text_a_list)

            if set_type == "test":
                label = "true"
            else:
                label = line['label']

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    @staticmethod
    def read_label(input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                data_json = json.loads(line[0])
                if data_json['label'] == "false":
                    labels.append(1)
                else:
                    labels.append(0)
            return labels
