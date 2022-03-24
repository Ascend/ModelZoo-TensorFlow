# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import numpy as np

from model.config import Config
from model.data_utils import CoNLLDataset, minibatches, pad_sequences


def data_to_bin(config):
    """Read data and convert it to bin"""

    # load test set
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    for idx, (words, labels) in enumerate(minibatches(test, config.batch_size)):
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0,
                                                   max_sequence_length=config.max_sequence_length,
                                                   max_word_length=config.max_word_length)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2,
                                               max_sequence_length=config.max_sequence_length,
                                               max_word_length=config.max_word_length)
        word_ids = np.array(word_ids)
        sequence_lengths = np.array(sequence_lengths)
        char_ids = np.array(char_ids)
        labels = np.array(labels)

        dir_bins = "./bin_data/"
        dir_word_ids = os.path.join(dir_bins, "word_ids/")
        dir_sequence_lengths = os.path.join(dir_bins, "sequence_lengths/")
        dir_char_ids = os.path.join(dir_bins, "char_ids/")
        dir_labels = os.path.join(dir_bins, "labels/")

        # create directories
        if not os.path.exists(dir_bins):
            os.mkdir(dir_bins)
        if not os.path.exists(dir_word_ids):
            os.mkdir(dir_word_ids)
        if not os.path.exists(dir_sequence_lengths):
            os.mkdir(dir_sequence_lengths)
        if not os.path.exists(dir_char_ids):
            os.mkdir(dir_char_ids)
        if not os.path.exists(dir_labels):
            os.mkdir(dir_labels)

        # store data as bin
        word_ids.tofile("{0}/{1:04d}.bin".format(dir_word_ids, idx))
        sequence_lengths.tofile("{0}/{1:04d}.bin".format(dir_sequence_lengths, idx))
        char_ids.tofile("{0}/{1:04d}.bin".format(dir_char_ids, idx))
        labels.tofile("{0}/{1:04d}.bin".format(dir_labels, idx))


if __name__ == '__main__':
    config = Config()
    config.batch_size = 1
    data_to_bin(config)
