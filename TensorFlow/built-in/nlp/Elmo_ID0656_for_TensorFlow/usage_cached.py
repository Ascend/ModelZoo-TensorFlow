#
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
#

'''
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''
from npu_bridge.npu_init import *

import os
import h5py
from bilm import dump_bilm_embeddings

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the dataset file.
dataset_file = 'dataset_file.txt'
with open(dataset_file, 'w') as fout:
    for sentence in tokenized_context + tokenized_question:
        fout.write(' '.join(sentence) + '\n')


# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
vocab_file = os.path.join(datadir, 'vocab_test.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Dump the embeddings to a file. Run this once for your dataset.
embedding_file = 'elmo_embeddings.hdf5'
dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file
)

# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(embedding_file, 'r') as fin:
    second_sentence_embeddings = fin['1'][...]

