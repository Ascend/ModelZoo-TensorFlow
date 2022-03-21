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

from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.vocab_file, max_word_length)

    test_prefix = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', type=str, default='ckpt_npu/', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', type=str, default='vocab/cn_vocab', help='Vocabulary file')
    parser.add_argument('--test_prefix', type=str, default='data/*', help='Prefix for test files')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    args = parser.parse_args()
    main(args)

