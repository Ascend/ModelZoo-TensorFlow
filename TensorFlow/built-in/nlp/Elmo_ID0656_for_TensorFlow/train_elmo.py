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


from npu_bridge.npu_init import *
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    max_word_len = 20
    vocab = load_vocab(args.vocab_file, max_word_len)

    # define the options
    # batch size for each GPU
    b_s = args.batch_size
    epoch_num = args.n_epochs
    batch_stop_num = args.n_batch_stop
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 999999999

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': max_word_len,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 100,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': epoch_num,
     'n_batch_stop': batch_stop_num,
     'n_train_tokens': n_train_tokens,
     'batch_size': b_s,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="ckpt_npu/", help='Location of checkpoint files')
    parser.add_argument('--vocab_file', default="vocab/cn_vocab", help='Vocabulary file')
    parser.add_argument('--train_prefix', default="data/*",help='Prefix for train files')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--n_epochs', type=int, help='n_epochs', default=10)
    parser.add_argument('--n_batch_stop', type=int, help='n_batch_stop', default=4000000)
    args = parser.parse_args()
    main(args)
