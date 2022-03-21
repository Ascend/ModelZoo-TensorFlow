# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
tf.app.flags.DEFINE_float("epoch", 7.5, "train epoch.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 160000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 80000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 12000000,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_compute_bleu", 10000,
                            "How many training steps to calculate the Bleu average score of verification set")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("compute_bleu_dev_set", False,
                            "Translate the validation set and calculate the Bleu average score")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("use_lstm", True,
                            "If true, we use LSTM cells instead of GRU cells.")
tf.app.flags.DEFINE_boolean("use_rev_sou_sen", False,
                            "If true, we use the training set of inverted source sentences.")

FLAGS = tf.app.flags.FLAGS

os.environ["ASCEND_DEVICE_ID"] = "4"

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# Calculate some training constants
iterations = int(FLAGS.max_train_data_size / FLAGS.batch_size)
epoch = FLAGS.epoch
total_step = int(epoch * iterations)
half_epoch_step = int(iterations / 2)

tf_config = tf.ConfigProto()
custom_op = tf_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
tf_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Remap must be explicitly turned off
tf_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF     # Must be explicitly closed


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


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


def train():
    """Train a en->fr translation model using WMT data."""
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None
    if FLAGS.from_train_data and FLAGS.to_train_data:
        from_train_data = FLAGS.from_train_data
        to_train_data = FLAGS.to_train_data
        from_dev_data = from_train_data
        to_dev_data = to_train_data
        if FLAGS.from_dev_data and FLAGS.to_dev_data:
            from_dev_data = FLAGS.from_dev_data
            to_dev_data = FLAGS.to_dev_data
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
            FLAGS.data_dir,
            from_train_data,
            to_train_data,
            from_dev_data,
            to_dev_data,
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size)
    else:
        # Prepare WMT data.
        print("Preparing WMT data in %s" % FLAGS.data_dir)
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size, FLAGS.use_rev_sou_sen)
    # print("batch_size:", FLAGS.batch_size)
    with tf.Session(config=tf_config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # sess.graph_def: graph description
        # tf.io.write_graph(sess.graph_def, FLAGS.train_dir, 'graph.pbtxt')

        # sess.graph: The graph put into session
        # writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(from_dev, to_dev)
        train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = model.global_step.eval()
        previous_losses = []
        current_epoch = current_step / iterations
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # # collect 1 iterations dump data
            # if current_step == 1:
            #     print("collect 1 iterations dump data current_step: %d" % current_step)
            #     break

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #     sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

            # Half epoch training is over
            if current_step % half_epoch_step == 0:
                current_epoch += 0.5
                if current_epoch < 5.0:
                    print(
                        "After the training of epoch %.1f, the training of epoch %.1f will begin. The current learning rate %.4f"
                        % (current_epoch, current_epoch + 0.5, model.learning_rate.eval()))
                else:
                    sess.run(model.learning_rate_decay_op)
                    print(
                        "After the training of epoch %.1f, start the training of epoch %.1f, half of the learning rate, the current learning rate %.4f"
                        % (current_epoch, current_epoch + 0.5, model.learning_rate.eval()))

            # All steps are finished, and the training is over
            if current_step == total_step:
                print("The %.1f epoch training is over, the training is over" % epoch)
                break

            # collect profiling and dump data
            # if current_step == 6:
            #     print("collect profiling and dump data current_step: %d" % current_step)
            #     break

    # writer.close()


# The translation verification set model is used to translate the original French text for Perl script to calculate Bleu score
def save_candidates_file(fr_sentence_candidate_list):
    candidatesFilePath = './newstest2013_candidate.fr'
    with open(candidatesFilePath, 'w', encoding='utf-8') as f:
        for fr_sentence_candidate in fr_sentence_candidate_list:
            f.write(fr_sentence_candidate + '\n')


# Translate a sentence
def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.from_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.to" % FLAGS.to_vocab_size)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)

            print(len(output_logits))
            print(np.array(output_logits).shape)
            print(output_logits)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


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


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.global_variables_initializer())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


def main(_):
    if FLAGS.self_test:
        print("----------------self_test-----------------")
        self_test()
    elif FLAGS.decode:
        print("----------------decode-----------------")
        decode()
    elif FLAGS.compute_bleu_dev_set:
        print("----------------compute_bleu_dev_set-----------------")
        compute_bleu_dev_set()
    else:
        print("----------------train-----------------")
        train()


if __name__ == "__main__":
    tf.app.run()
