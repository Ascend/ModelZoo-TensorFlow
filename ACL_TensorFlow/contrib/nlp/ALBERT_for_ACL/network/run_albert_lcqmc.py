# -*- coding: utf-8 -*-
# @Author: fengtingyan
# @Date:   2023-3-29 21:11:00
# @Last Modified by:   fengtingyan
# @Last Modified time: 2023-3-29 21:11:00

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

__all__ = 'LcqmcProcessor'


class LcqmcProcessor(DataProcessor):
    """Processor for the AFQMC data set."""

    def get_examples(self):
        """See base class."""
        return self.create_examples(self.read_json(os.path.join(FLAGS.data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        #todo
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line['sentence1'])
            text_b = tokenization.convert_to_unicode(line['sentence2'])
            label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else '0'
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def read_label(input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                data_json = json.loads(line[0])
                labels.append(int(data_json['label']))
            return labels
