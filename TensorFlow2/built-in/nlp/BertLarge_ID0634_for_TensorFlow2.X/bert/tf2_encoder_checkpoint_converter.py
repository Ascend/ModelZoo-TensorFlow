# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A converter from a V1 BERT encoder checkpoint to a V2 encoder checkpoint.

The conversion will yield an object-oriented checkpoint that can be used
to restore a TransformerEncoder object.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from tf2_common.modeling import activations
import configs
import tf1_checkpoint_converter_lib
from modeling import networks
from modeling.networks import bert_pretrainer
import tf2_common.modeling.tf_utils as tf_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string(
    "checkpoint_to_convert", None,
    "Initial checkpoint from a pretrained BERT model core (that is, only the "
    "BertModel, with no task heads.)")
flags.DEFINE_string("converted_checkpoint_path", None,
                    "Name for the created object-based V2 checkpoint.")

flags.DEFINE_boolean(name='use_packed_model', default=True,
                    help="whether to enable packed model, default is True.")

flags.DEFINE_integer(
    "max_sequences_per_pack", 3, 
    "Maximum number of sequences per sequence. "
    "Must match data generation.")
flags.DEFINE_float(
    "average_sequences_per_sample", 1.999, 
    "average number of sequences per sample. "
    "Must match data generation.")
flags.DEFINE_boolean(name='use_fastgelu', default=True,
                    help='whether to enable fastgelu, default is True.')
flags.DEFINE_integer('max_predictions_per_seq', 20, 
                     'Maximum predictions per sequence_output. ')

def _create_bert_model(cfg):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    cfg: A `BertConfig` to create the core model.
  Returns:
    A keras model.
  """
  bert_encoder = networks.TransformerEncoder(
      vocab_size=cfg.vocab_size,
      hidden_size=cfg.hidden_size,
      num_layers=cfg.num_hidden_layers,
      num_attention_heads=cfg.num_attention_heads,
      intermediate_size=cfg.intermediate_size,
      activation=activations.gelu,
      dropout_rate=cfg.hidden_dropout_prob,
      attention_dropout_rate=cfg.attention_probs_dropout_prob,
      sequence_length=cfg.max_position_embeddings,
      type_vocab_size=cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=cfg.initializer_range))
  max_predictions_per_seq = FLAGS.max_predictions_per_seq

  initializer = tf.keras.initializers.TruncatedNormal(
        stddev=cfg.initializer_range)
  pretrainer_model = bert_pretrainer.BertPretrainer(
      network=bert_encoder,
      num_classes=2, # The next sentence prediction label has two classes.
      num_token_predictions=max_predictions_per_seq,
      activation=tf_utils.get_activation(cfg.hidden_act),
      initializer=initializer,
      output='predictions')
  return pretrainer_model

  #return bert_encoder


def convert_checkpoint(bert_config, output_path, v1_checkpoint):
  """Converts a V1 checkpoint into an OO V2 checkpoint."""
  output_dir, _ = os.path.split(output_path)

  # Create a temporary V1 name-converted checkpoint in the output directory.
  temporary_checkpoint_dir = os.path.join(output_dir, "temp_v1")
  temporary_checkpoint = os.path.join(temporary_checkpoint_dir, "ckpt")
  tf1_checkpoint_converter_lib.convert(
      checkpoint_from_path=v1_checkpoint,
      checkpoint_to_path=temporary_checkpoint,
      num_heads=bert_config.num_attention_heads,
      name_replacements=tf1_checkpoint_converter_lib.BERT_V2_NAME_REPLACEMENTS,
      permutations=tf1_checkpoint_converter_lib.BERT_V2_PERMUTATIONS,
      exclude_patterns=["lamb", "Lamb"])

  # Create a V2 checkpoint from the temporary checkpoint.
  model = _create_bert_model(bert_config)
  tf1_checkpoint_converter_lib.create_v2_checkpoint(model, temporary_checkpoint,
                                                    output_path)

  # Clean up the temporary checkpoint, if it exists.
  try:
    tf.io.gfile.rmtree(temporary_checkpoint_dir)
  except tf.errors.OpError:
    # If it doesn't exist, we don't need to clean it up; continue.
    pass


def main(_):
  tf.enable_v2_behavior()
  output_path = FLAGS.converted_checkpoint_path
  v1_checkpoint = FLAGS.checkpoint_to_convert
  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  convert_checkpoint(bert_config, output_path, v1_checkpoint)


if __name__ == "__main__":
  app.run(main)
