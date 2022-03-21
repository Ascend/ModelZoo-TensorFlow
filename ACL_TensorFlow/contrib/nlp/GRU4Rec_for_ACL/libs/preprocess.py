# coding=utf-8
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

import os

from absl import flags

from network import fine_tuning_utils

FLAGS = flags.FLAGS


def preprocess(processor):
    """Preprocess func is used to convert txt files to acl input bin files"""
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if FLAGS.model_name.lower() in ("lstm", "transformer", "gru"):
        processor.get_features(True)
    else:
        label_list = processor.get_labels()

        tokenizer = fine_tuning_utils.create_vocab()

        examples = processor.get_examples()

        processor.convert_examples_to_features(examples, tokenizer, label_list, True)
