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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import tensorflow.compat.v1 as tf
from absl import flags

from network.classifier_utils import DataProcessor, InputExample

FLAGS = flags.FLAGS

__all__ = 'LcqmcProcessor'


class LcqmcProcessor(DataProcessor):
    """Processor for the LCQMC data set (GLUE version)."""

    def get_examples(self):
        """See base class."""
        return self.create_examples(self.read_tsv(os.path.join(FLAGS.data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        print("length of lines:", len(lines))
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = self.process_text(line[2])
            text_a = self.process_text(line[0])
            text_b = self.process_text(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def read_label(input_file):
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            labels = []
            for (i, line) in enumerate(reader):
                labels.append(int(line[2]))
            return labels
