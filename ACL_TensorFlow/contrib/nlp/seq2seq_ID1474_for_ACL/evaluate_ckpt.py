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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model
import compute_bleu

tf.app.flags.DEFINE_float("learning_rate", 0.7, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1000, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 160000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 80000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_lstm", True,
                            "If true, we use LSTM cells instead of GRU cells.")

FLAGS = tf.app.flags.FLAGS

os.environ["ASCEND_DEVICE_ID"] = "1"

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

tf_config = tf.ConfigProto()
custom_op = tf_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
tf_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
tf_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        use_lstm=FLAGS.use_lstm,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


# The translation verification set model is used to translate the original French text for Perl script to calculate Bleu score
def save_candidates_file(fr_sentence_candidate_list):
    candidatesFilePath = './newstest2013_candidate.fr'
    with open(candidatesFilePath, 'w', encoding='utf-8') as f:
        for fr_sentence_candidate in fr_sentence_candidate_list:
            f.write(fr_sentence_candidate + '\n')


# Translate the validation set and calculate the Bleu average score
def compute_bleu_dev_set():
    dev_set_str_list = data_utils.extract_dev_set(FLAGS.data_dir)
    bleu = compute_bleu.Bleu()
    bleu_score_list = []
    fr_sentence_candidate_list = []
    with tf.Session(config=tf_config) as sess_bleu:
        # Create model and load parameters.
        model = create_model(sess_bleu, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.from_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.to" % FLAGS.to_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
        for dev_set_str in dev_set_str_list:
            en_sentence = dev_set_str[0]
            fr_sentence_references = [dev_set_str[1].split()]
            # print(fr_sentence_references)

            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(en_sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", en_sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess_bleu, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # This is a beam search decoder
            # outputs = beam_search(output_logits, 12)

            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            fr_sentence_candidate = [tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]
            # print(fr_sentence_candidate)
            # print(" ".join(fr_sentence_candidate))
            # Save the model translation sentences to the list
            fr_sentence_candidate_list.append(" ".join(fr_sentence_candidate))
            # Calculate and save the Bleu scores of predicted and real sentences
            bleu_score_list.append(bleu.score(fr_sentence_references, fr_sentence_candidate))
            print("The %d sentence is calculated" % len(bleu_score_list))
    save_candidates_file(fr_sentence_candidate_list)
    bleu_score_avg = np.mean(bleu_score_list)
    print("The average score of Bleu : ", bleu_score_avg)
    print("The average score of Bleu*100 : ", bleu_score_avg * 100)


def main(_):
    compute_bleu_dev_set()


if __name__ == "__main__":
    tf.app.run()
