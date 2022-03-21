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

import csv
import os

import tensorflow.compat.v1 as tf
from absl import flags

from network import tokenization
from network.classifier_utils import DataProcessor, InputExample

FLAGS = flags.FLAGS

__all__ = 'ReProcessor'


class ReProcessor(DataProcessor):

    def get_examples(self):
        return self.create_example(self.read_tsv(os.path.join(FLAGS.data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def create_example(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[2])
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    @staticmethod
    def read_label(input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                labels.append(int(line[2]))
            return labels
