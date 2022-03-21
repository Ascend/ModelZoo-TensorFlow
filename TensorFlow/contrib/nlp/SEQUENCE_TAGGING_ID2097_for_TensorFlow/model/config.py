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
import argparse

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="./data", help="Dataset path.")
    parser.add_argument("--output_path", type=str, default="./results/", help="Output path.")
    parser.add_argument("--resume", type=int, default=0, help="Resume training.")
    parser.add_argument("--dir_ckpt", type=str, default="./results/model.weights/",
                        help="Checkpoint path for evaluation.")
    parser.add_argument("--dim_word", type=int, default=100, help="The dimension of word embeddings.")
    parser.add_argument("--dim_char", type=int, default=30, help="The dimension of char embeddings.")
    parser.add_argument("--max_sequence_length", type=int, default=128, help="Max length of sequence.")
    parser.add_argument("--max_word_length", type=int, default=64, help="Max length of word.")
    parser.add_argument("--train_embeddings", type=bool, default=False, help="Whether to train embeddings.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout.")
    parser.add_argument("--batchsize", type=int, default=10, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=('adam', 'sgd', 'momentum', 'adagrad', 'rmsprop'), help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument("--lr_decay", type=float, default=0.05, help="The decay rate of learning rate.")
    parser.add_argument("--grad_clip", type=float, default=10, help="Gradient clip.")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stop.")
    parser.add_argument("--mix_precision", type=bool, default=False, help="Whether to enable mix precision.")
    parser.add_argument("--hidden_size_lstm", type=int, default=200, help="Hidden size of lstm.")
    parser.add_argument("--use_crf", type=bool, default=True, help="Whether to use crf.")
    parser.add_argument("--use_chars", type=bool, default=True, help="Whether to use char embeddings.")
    parser.add_argument("--conv_kernel_size", type=int, default=3, help="Kernel size of cnn block.")
    parser.add_argument("--conv_filter_num", type=int, default=30, help="Filter number of cnn block.")
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    # general config
    dir_ckpt = args.dir_ckpt
    if dir_ckpt[-1] != '/':
        dir_ckpt += '/'
    dir_model = os.path.join(output_path, "model.weights/")
    path_log = os.path.join(output_path, "log.txt")

    # embeddings
    dim_word = args.dim_word
    dim_char = args.dim_char

    max_sequence_length = args.max_sequence_length
    max_word_length = args.max_word_length

    # glove files
    filename_glove = os.path.join(data_path, "glove.6B/glove.6B.{}d.txt".format(dim_word))
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = os.path.join(data_path, "glove.6B.{}d.trimmed.npz".format(dim_word))
    use_pretrained = True

    # dataset
    filename_dev = os.path.join(data_path, "coNLL/eng/eng.testa.iob")
    filename_test = os.path.join(data_path, "coNLL/eng/eng.testb.iob")
    filename_train = os.path.join(data_path, "coNLL/eng/eng.train.iob")

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = os.path.join(data_path, "words.txt")
    filename_tags = os.path.join(data_path, "tags.txt")
    filename_chars = os.path.join(data_path, "chars.txt")

    # resume
    resume = args.resume

    # training
    train_embeddings = args.train_embeddings
    nepochs = args.epochs
    dropout = args.dropout
    batch_size = args.batchsize
    lr_method = args.optimizer
    lr = args.lr
    lr_decay = args.lr_decay
    clip = args.grad_clip
    nepoch_no_imprv = args.early_stop

    # model hyperparameters
    hidden_size_lstm = args.hidden_size_lstm  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = args.use_crf  # if crf, training is 1.7x slower on CPU
    use_chars = args.use_chars  # if char embedding, training is 3.5x slower on CPU
    conv_kernel_size = args.conv_kernel_size
    conv_filter_num = args.conv_filter_num
