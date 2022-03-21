# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python3

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

"""Helper library for ALBERT fine-tuning.

This library can be used to construct ALBERT models for fine-tuning, either from
json config files or from TF-Hub modules.
"""

# delete tf hub

from absl import flags

from network import tokenization

FLAGS = flags.FLAGS


def create_bert(input_ids, input_mask, segment_ids):
    # delete tf hub
    """Creates an ALBERT, either from TF-Hub or from scratch."""

    if FLAGS.model_name == "albert_en":
        from network.albert_en import modeling

    elif FLAGS.model_name == "albert_zh":
        from network.albert_zh import modeling

    else:
        from network.bert import modeling

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    model = modeling.BertModel(
        config=bert_config,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids
    )

    return model.get_pooled_output(), model.get_sequence_output()


def create_vocab():
    # delete tf hub
    """Creates a vocab, either from vocab file or from a TF-Hub module."""
    return tokenization.FullTokenizer.from_scratch(FLAGS.vocab_file, FLAGS.do_lower_case, FLAGS.spm_model_file)
