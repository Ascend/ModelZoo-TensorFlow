# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
#"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf

####################NPU_modify start####################
import time
from utils.utils import LogEvalRunHook
from gpu_environment import get_custom_getter

from npu_bridge.estimator.npu.npu_config import *
from npu_bridge.estimator.npu.npu_estimator import *
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

os.environ['WHICH_OP'] = 'GEOP'
os.environ['NEW_GE_FE_ID'] = '1'
os.environ['GE_AICPU_FLAG'] = '1'
os.environ['GE_USE_STATIC_MEMORY'] = '1'
os.environ['OPTION_EXEC_HCCL_FLAG'] = '1'
os.environ['HCCL_CONNECT_TIMEOUT'] = '600'
####################NPU_modify end######################

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

####################NPU_modify start####################
flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU. "
                                        "Manual casting is done instead of using AMP")

flags.DEFINE_bool("use_fp16", False, "Whether to enable AMP ops.")

flags.DEFINE_integer("display_loss_steps", 10, "How often to print loss")

flags.DEFINE_bool("report_loss", True, "Whether to report total loss during training.")

flags.DEFINE_bool("distributed", False, "Whether to use multi-npu")

flags.DEFINE_bool("use_fp16_cls", False, "Whether to use fp16 in cls and pooler.")

flags.DEFINE_bool('npu_bert_fused_gelu', True, "Whether to use npu defined gelu op.")

flags.DEFINE_bool('npu_bert_debug', False, "If True, dropout and shuffle is disable.")

flags.DEFINE_bool('npu_bert_npu_dropout', True, 'Whether to use npu defined gelu op')

flags.DEFINE_bool('customer_bert_gather', False, 'Whether to use customer defined gather op')

flags.DEFINE_integer("npu_bert_loss_scale", -1, "Whether to use loss scale, -1 is disable, 0 is dynamic loss scale, >=1 is static loss scale")

flags.DEFINE_integer('init_loss_scale_value', 2**32, 'Initial loss scale value for loss scale optimizer')

flags.DEFINE_bool("npu_bert_clip_by_global_norm", True, "Use clip_by_global_norm if True, or use clip_by_norm for each gradient")

flags.DEFINE_bool('npu_bert_use_tdt', True, 'Whether to use tdt as dataset')

flags.DEFINE_bool('hcom_parallel', True, 'Whether to use parallel allreduce')

class _LogSessionRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, display_every=10):
    self.global_batch_size = global_batch_size
    self.display_every = display_every
  def after_create_session(self, session, coord):
    self.elapsed_secs = 0.
    self.count = 0
    self.all_count = 0
    self.avg_loss = 0.0

  def before_run(self, run_context):
    self.t0 = time.time()

    if (tf.flags.FLAGS.npu_bert_loss_scale == 0) and (FLAGS.manual_fp16 or FLAGS.use_fp16):
      return tf.train.SessionRunArgs(
        fetches=['global_step:0', 'total_loss:0',
                 'learning_rate:0', 'nsp_loss:0',
                 'mlm_loss:0', 'loss_scale:0', 'apply_grads/overflow_status_reduce_all:0 '])
    else:
      return tf.train.SessionRunArgs(
        fetches=['global_step:0', 'total_loss:0',
                 'learning_rate:0', 'nsp_loss:0',
                 'mlm_loss:0'])

  def after_run(self, run_context, run_values):
    self.elapsed_secs += time.time() - self.t0

    if (tf.flags.FLAGS.npu_bert_loss_scale == 0) and (FLAGS.manual_fp16 or FLAGS.use_fp16):
        global_step, total_loss, lr, nsp_loss, mlm_loss, loss_scaler, custom_arg = run_values.results
    else:
        global_step, total_loss, lr, nsp_loss, mlm_loss = run_values. \
            results
    update_step = True

    print_step = global_step + 1 # One-based index for printing.
    self.avg_loss += total_loss
    self.all_count += 1
    if update_step:
        self.count += 1
        dt = self.elapsed_secs / self.count
        sent_per_sec = self.global_batch_size / dt * FLAGS.iterations_per_loop
        avg_loss_step = self.avg_loss / self.all_count

        if (tf.flags.FLAGS.npu_bert_loss_scale == 0) and (FLAGS.manual_fp16 or FLAGS.use_fp16):
          print('Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %9.6f Average Loss = %9.6f LR = %6.4e Loss scale = %6.4e isFinite = %6i' %
                (print_step, sent_per_sec, mlm_loss, nsp_loss, total_loss, avg_loss_step, lr, loss_scaler, custom_arg), flush=True)
        else:
          print('Step = %6i Throughput = %11.1f MLM Loss = %10.4e NSP Loss = %10.4e Loss = %9.6f Average Loss = %9.6f LR = %6.4e' %
                (print_step, sent_per_sec, mlm_loss, nsp_loss, total_loss, avg_loss_step, lr), flush=True)
        self.elapsed_secs = 0.
        self.count = 0
        self.avg_loss = 0.0
        self.all_count = 0

####################NPU_modify end######################

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    ####################NPU_modify start####################
    masked_lm_loss = tf.identity(masked_lm_loss, name="mlm_loss")
    next_sentence_loss = tf.identity(next_sentence_loss, name="nsp_loss")
    ####################NPU_modify end######################

    total_loss = masked_lm_loss + next_sentence_loss

    ####################NPU_modify start####################
    total_loss = tf.identity(total_loss, name='total_loss')
    ####################NPU_modify end######################

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, FLAGS.manual_fp16, use_tpu)

      ####################NPU_modify start####################
      if not use_tpu:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op)
      else:
      ####################NPU_modify end######################
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform", custom_getter=get_custom_getter(compute_type=tf.float16 if FLAGS.use_fp16_cls else tf.float32)):
      ####################NPU_modify start####################
      if FLAGS.use_fp16_cls:
        input_tensor = tf.cast(input_tensor, tf.float16)
      ####################NPU_modify end######################
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      ####################NPU_modify start####################
      input_tensor = tf.cast(input_tensor, tf.float32)
      ####################NPU_modify end######################
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    ####################NPU_modify start####################
    if FLAGS.use_fp16_cls:
      input_tensor = tf.cast(input_tensor, tf.float16)
      logits = tf.matmul(input_tensor, tf.cast(output_weights, tf.float16), transpose_b=True)
      logits = tf.cast(logits, tf.float32)
    else:
    ####################NPU_modify end######################
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)

    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    ####################NPU_modify start####################
    if FLAGS.use_fp16_cls:
      input_tensor = tf.cast(input_tensor, tf.float16)
      logits = tf.matmul(input_tensor, tf.cast(output_weights, tf.float16), transpose_b=True)
      logits = tf.cast(logits, tf.float32)
    else:
    ####################NPU_modify end######################
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      ####################NPU_modify start####################
      if FLAGS.distributed:
          rank_size = int(os.getenv('RANK_SIZE'))
          rank_index = int(os.getenv('RANK_INDEX'))
          device_id = int(os.getenv('DEVICE_ID'))
          local_rank = device_id + rank_index * 8
          tf.logging.info("RANK_SIZE=%d, local_rank=%d", rank_size, local_rank)
          d = d.shard(rank_size, local_rank)
      ####################NPU_modify end######################
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      ####################NPU_modify start####################
      if not FLAGS.npu_bert_debug:
        cycle_length = min(num_cpu_threads, int(len(input_files) / int(os.getenv('RANK_SIZE'))))
      else:
        cycle_length = 1
      ####################NPU_modify end######################

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      #######change tf.data.experimental.parallel_interleave by jwx644041######
      #d = d.apply(
          #tf.contrib.data.parallel_interleave(
           #   tf.data.TFRecordDataset,
           #   sloppy=is_training,
           #   cycle_length=cycle_length))
      d = d.interleave(
          tf.data.TFRecordDataset,
          cycle_length=cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      #######change tf.data.experimental.parallel_interleave by jwx644041######
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  ####################NPU_modify start####################
  if FLAGS.use_fp16:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  ####################NPU_modify end######################

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
  ####################NPU_modify start####################
    input_files.extend(tf.gfile.Glob(os.path.join(input_pattern, "*")))

  input_files.sort()
  ####################NPU_modify end######################


  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  ####################NPU_modify start####################
  config = tf.ConfigProto()

  run_config = NPURunConfig(
      model_dir=FLAGS.output_dir,
      save_summary_steps=0,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=1 if FLAGS.report_loss else 100,
      enable_data_pre_proc=FLAGS.npu_bert_use_tdt,
      iterations_per_loop=FLAGS.iterations_per_loop,
      hcom_parallel=FLAGS.hcom_parallel)

  if FLAGS.distributed:
      rank_size = int(os.getenv('RANK_SIZE'))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate if not (FLAGS.distributed) else FLAGS.learning_rate*rank_size,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  training_hooks = []

  if FLAGS.report_loss:
      global_batch_size = FLAGS.train_batch_size if not FLAGS.distributed else FLAGS.train_batch_size * rank_size
      training_hooks.append(
          _LogSessionRunHook(global_batch_size, FLAGS.display_loss_steps))


  estimator = NPUEstimator(model_fn=model_fn, config=run_config)
  ####################NPU_modify end######################

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.train_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.eval_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
