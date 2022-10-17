# Lint as: python3
# coding=utf-8
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
# ==============================================================================
"""Training script for DELF/G on Google Landmarks Dataset.

Uses classification loss, with MirroredStrategy, to support running on multiple
GPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os
import time
import itertools
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
tf.disable_eager_execution()
import numpy as np

# Placeholder for internal import. Do not remove this line.
from datasets.google_landmarks_dataset import googlelandmarks as gld
from model import delf_model


FLAGS = flags.FLAGS
flags.DEFINE_string('output_path', '/tmp/delf', 'WithTensorBoard output_path.')
flags.DEFINE_string('data_path', '/tmp/data', 'data path.')
flags.DEFINE_enum(
    'dataset_version', 'gld_v1', ['gld_v1', 'gld_v2', 'gld_v2_clean'],
    'Google Landmarks dataset version, used to determine the number of '
    'classes.')
flags.DEFINE_integer('seed', 0, 'Seed to training dataset.')
flags.DEFINE_float('initial_lr', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
flags.DEFINE_integer('max_iters', 500000, 'Maximum iterations.')
flags.DEFINE_boolean('block3_strides', True, 'Whether to use block3_strides.')
flags.DEFINE_boolean('use_augmentation', True,
                     'Whether to use ImageNet style augmentation.')
flags.DEFINE_float(
    'attention_loss_weight', 1.0,
    'Weight to apply to the attention loss when calculating the '
    'total loss of the model.')
flags.DEFINE_boolean('delg_global_features', False,
                     'Whether to train a DELG model.')
flags.DEFINE_float(
    'delg_gem_power', 3.0, 'Power for Generalized Mean pooling. Used only if '
    'delg_global_features=True.')
flags.DEFINE_integer(
    'delg_embedding_layer_dim', 2048,
    'Size of the FC whitening layer (embedding layer). Used only if'
    'delg_global_features:True.')
flags.DEFINE_float(
    'delg_scale_factor_init', 45.25,
    'Initial value of the scaling factor of the cosine logits. The default '
    'value is sqrt(2048). Used only if delg_global_features=True.')
flags.DEFINE_float('delg_arcface_margin', 0.1,
                   'ArcFace margin. Used only if delg_global_features=True.')
flags.DEFINE_integer('image_size', 321, 'Size of each image side to use.')
flags.DEFINE_boolean('use_autoencoder', True,
                     'Whether to train an autoencoder.')
flags.DEFINE_float(
    'reconstruction_loss_weight', 10.0,
    'Weight to apply to the reconstruction loss from the autoencoder when'
    'calculating total loss of the model. Used only if use_autoencoder=True.')
flags.DEFINE_integer(
    'autoencoder_dimensions', 128,
    'Number of dimensions of the autoencoder. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_integer(
    'local_feature_map_channels', 1024,
    'Number of channels at backbone layer used for local feature extraction. '
    'Default value 1024 is the number of channels of block3. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_boolean('load_checkpoint', False, 'whether load checkpoint.')
flags.DEFINE_integer(
    'report_interval', 500,
    'log for training')
flags.DEFINE_integer(
    'eval_interval', 1000,
    'log for eval')
flags.DEFINE_integer(
    'save_interval', 10000,
    'log for saving model')


def _learning_rate_schedule(global_step, max_steps, initial_lr):
  """Calculates learning_rate with linear decay.
  Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.
  Returns:
    lr: float, learning rate.
  """
  global_step_value = tf.cast(global_step, tf.float32)
  lr = initial_lr * (1.0 - global_step_value / float(max_steps))
  return lr


def train_step(model, images, labels, global_batch_size, max_steps, FLAGS):
  # Temporary workaround to avoid some corrupted labels.
  labels = tf.clip_by_value(labels, 0, model.num_classes)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = _learning_rate_schedule(global_step, max_steps, FLAGS.initial_lr)
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True, use_locking=True)
  ## added for enabling loss scale and mix precision
  #optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
  loss_scale_manager = FixedLossScaleManager(loss_scale=100, enable_overflow_check=False)
  # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
  #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
  optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)

  # Compute loss.
  # Set reduction to `none` so we can do the reduction afterwards and divide
  # by global batch size.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)

  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=global_batch_size)

  # Make a forward pass to calculate prelogits.
  (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
    dim_expanded_features, _) = model.global_and_local_forward_pass(images)

  # Calculate global loss by applying the descriptor classifier.
  desc_logits = model.desc_classification(desc_prelogits)
  desc_loss = compute_loss(labels, desc_logits)

  # Calculate attention loss by applying the attention block classifier.
  attn_logits = model.attn_classification(attn_prelogits)
  attn_loss = compute_loss(labels, attn_logits)

  # Calculate reconstruction loss between the attention prelogits and the backbone.
  block3 = tf.stop_gradient(backbone_blocks['block3'])
  reconstruction_loss = tf.math.reduce_mean(tf.keras.losses.MSE(block3, dim_expanded_features))

  # Cumulate global loss, attention loss and reconstruction loss.
  total_loss = (
      desc_loss + FLAGS.attention_loss_weight * attn_loss +
      FLAGS.reconstruction_loss_weight * reconstruction_loss)

  # TODO: Apply gradient
  variables = tf.trainable_variables()
  gradients = tf.gradients(total_loss, variables)
  clip_val = tf.constant(10.0)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
  update_ops = list(itertools.chain(*tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
  with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(zip(clipped, variables),
                                         global_step=global_step)
  # Input image-related summaries.
  ## tf.summary.image('batch_images', (images + 1.0) / 2.0)
  ## tf.summary.scalar(
  ##     'image_range/max', tf.reduce_max(images))
  ## tf.summary.scalar(
  ##     'image_range/min', tf.reduce_min(images))

  # Record statistics of the attention score
  ## tf.summary.image('batch_attention', attn_scores / tf.reduce_max(attn_scores + 1e-3))
  ## tf.summary.scalar('attention_score/max', tf.reduce_max(attn_scores))
  ## tf.summary.scalar('attention_score/min', tf.reduce_min(attn_scores))
  tf.summary.scalar('attention_score/mean', tf.reduce_mean(attn_scores))

  ## Record desc accuracy
  desc_accuracy = tf.equal(tf.argmax(desc_logits, 1), labels)
  desc_accuracy = tf.reduce_mean(tf.cast(desc_accuracy, tf.float32))
  tf.summary.scalar('train_accuracy/desc', desc_accuracy)
  # desc_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='desc_train_accuracy')
  # desc_softmax_probabilities = tf.keras.layers.Softmax()(desc_logits)
  # desc_train_accuracy.update_state(labels, desc_softmax_probabilities)
  # tf.summary.scalar('desc_train_accuracy', desc_train_accuracy.result())

  ## Record attn accuracy
  attn_accuracy = tf.equal(tf.argmax(attn_logits, 1), labels)
  attn_accuracy= tf.reduce_mean(tf.cast(attn_accuracy, tf.float32))
  tf.summary.scalar('train_accuracy/attn', attn_accuracy)
  # attn_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='attn_train_accuracy')
  # attn_softmax_probabilities = tf.keras.layers.Softmax()(attn_logits)
  # attn_train_accuracy.update_state(labels, attn_softmax_probabilities)
  # tf.summary.scalar('attn_train_accuracy', attn_train_accuracy.result())

  ## Record losses
  tf.summary.scalar('train_loss/desc', desc_loss)
  tf.summary.scalar('train_loss/attn', attn_loss)
  tf.summary.scalar('train_loss/reconstruction', reconstruction_loss)
  tf.summary.scalar('train_loss/total', total_loss)
  merged_summary_op = tf.summary.merge_all()
  return merged_summary_op, train_op, desc_loss, attn_loss, reconstruction_loss


def eval_step(model, images, labels):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)
  labels = tf.clip_by_value(labels, 0, model.num_classes)

  # descriptor predictions.
  blocks = {}
  prelogits = model.backbone(
      images, intermediates_dict=blocks, training=False)
  desc_logits = model.desc_classification(prelogits, training=False)
  desc_loss = loss_object(labels, desc_logits)

  desc_accuracy = tf.equal(tf.argmax(desc_logits, 1), labels)
  desc_accuracy = tf.reduce_mean(tf.cast(desc_accuracy, tf.float32))
  # desc_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='desc_validation_accuracy')
  # desc_softmax_probabilities = tf.keras.layers.Softmax()(desc_logits)
  # desc_validation_accuracy.update_state(labels, desc_softmax_probabilities)

  ## attention predictions
  block3 = blocks['block3']  # pytype: disable=key-error
  prelogits, _, _ = model.attention(block3, training=False)
  attn_logits = model.attn_classification(prelogits, training=False)
  attn_loss = loss_object(labels, attn_logits)
  attn_accuracy = tf.equal(tf.argmax(attn_logits, 1), labels)
  attn_accuracy = tf.reduce_mean(tf.cast(attn_accuracy, tf.float32))
  # attn_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='attn_validation_accuracy')
  # attn_softmax_probabilities = tf.keras.layers.Softmax()(attn_logits)
  # attn_validation_accuracy.update_state(labels, attn_softmax_probabilities)

  return desc_loss, desc_accuracy, attn_loss, attn_accuracy


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #-------------------------------------------------------------
  # Log flags used.
  logging.info('Running training script with\n')
  logging.info('data_path= %s', FLAGS.data_path)
  logging.info('output_path= %s', FLAGS.output_path)
  logging.info('initial_lr= %f', FLAGS.initial_lr)
  logging.info('block3_strides= %s', str(FLAGS.block3_strides))

  max_iters = FLAGS.max_iters
  global_batch_size = FLAGS.batch_size
  image_size = FLAGS.image_size
  eval_batch_size = global_batch_size * 1
  num_eval_batches = int(20000 / eval_batch_size)

  # Determine the number of classes based on the version of the dataset.
  gld_info = gld.GoogleLandmarksInfo()
  num_classes = gld_info.num_classes[FLAGS.dataset_version]

  tf.reset_default_graph()
  # Setup session
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config_proto.allow_soft_placement = True
  # added for enabling mix precision and loss scale
  # config_proto.graph_options.rewrite_options.auto_mixed_precision = 1

  ## use NpuOptimizer
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

  ## set mix precision
  # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  ### set black list for mix precision
  #custom_op.parameter_map['modify_mixlist'].s = tf.compat.as_bytes(OPS_FILE)
  #print("[PrecisionTool] Set mix_precision setting file: ", OPS_FILE)
  ### set fusion switch
  #custom_op.parameter_map['fusion_switch_file'].s = tf.compat.as_bytes(FUSION_SWITCH_FILE)
  #print("[PrecisionTool] Set fusion switch file: ", FUSION_SWITCH_FILE)
  ## enable dump
  #dump_data_dir = "/cache/dump_data"
  #custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_data_dir)
  #custom_op.parameter_map["enable_dump"].b = True
  #custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
  #custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")

  ## enable auto tune
  # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

  ### enable profiling mode
  ##profile_data_dir = "/cache/profile_data"
  ##custom_op.parameter_map["profiling_mode"].b = True
  ### training option
  ### profiling_options = '{"output":"%s","storage_limit": "300MB","training_trace":"on","task_trace":"on"}' % profile_data_dir
  ### test option
  ##profiling_options = '{"output":"%s","storage_limit": "300MB","task_trace":"on","aicpu":"on"}' % profile_data_dir
  ##logging.info(profiling_options)
  ##custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profiling_options)

  tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))

  # ------------------------------------------------------------
  # Create train/validation sets.
  train_file_pattern = os.path.join(FLAGS.data_path, "google-landmark/tfrecord/train*")
  train_dataset = gld.CreateDataset(
      file_pattern=train_file_pattern,
      batch_size=global_batch_size,
      image_size=image_size,
      augmentation=FLAGS.use_augmentation,
      seed=FLAGS.seed)
  train_data_inter = train_dataset.make_initializable_iterator()
  train_images, train_labels = train_data_inter.get_next()

  validation_file_pattern = os.path.join(FLAGS.data_path, "google-landmark/tfrecord/validation*")
  validation_dataset = gld.CreateDataset(
      file_pattern=validation_file_pattern,
      batch_size=eval_batch_size,
      image_size=image_size,
      augmentation=False,
      seed=FLAGS.seed)
  validation_data_inter = validation_dataset.make_initializable_iterator()
  val_images, val_labels = validation_data_inter.get_next()

  # ------------------------------------------------------------
  # Create train operation.
  model = delf_model.Delf(
      block3_strides=FLAGS.block3_strides,
      name='DELF',
      use_dim_reduction=FLAGS.use_autoencoder,
      reduced_dimension=FLAGS.autoencoder_dimensions,
      dim_expand_channels=FLAGS.local_feature_map_channels)
  model.init_classifiers(num_classes)
  merged_summary_op, train_op, desc_loss, attn_loss, reconstruction_loss = train_step(model, train_images, train_labels, global_batch_size, max_iters, FLAGS)

  # ------------------------------------------------------------
  # Create test operation.
  val_desc_loss, val_desc_acc, val_attn_loss, val_attn_acc = eval_step(model, val_images, val_labels)

  # ------------------------------------------------------------
  # Create summary.
  summary_dir = os.path.join(FLAGS.output_path, "summaries")
  summary_writer = tf.summary.FileWriter(summary_dir, tfs.graph)
  test_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output_path, "test_summaries"))

  # ------------------------------------------------------------
  # Create a checkpoint directory to store the checkpoints.
  saver = tf.train.Saver(max_to_keep=3)
  model_dir = os.path.join(FLAGS.output_path, "ckpts")
  model_path = os.path.join(model_dir, "model.ckpt")

  global_step_value = 0
  global_step = tf.train.get_or_create_global_step()
  tfs.run(tf.global_variables_initializer())
  tfs.run(tf.local_variables_initializer())
  tfs.run(train_data_inter.initializer)
  tfs.run(validation_data_inter.initializer)

  if FLAGS.load_checkpoint:
    ckpt_dir = os.path.join(FLAGS.data_path, "best_ckpts")
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(tfs, latest_ckpt)
    global_step_value = tfs.run(global_step)
    logging.info("Restore from ckpt:{}, global_step={}".format(latest_ckpt, global_step_value))
  else:
    imagenet_ckpt_dir = os.path.join(FLAGS.data_path, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    logging.info('Attempting to load ImageNet pretrained weights:{}'.format(imagenet_ckpt_dir))
    model.backbone.restore_weights(imagenet_ckpt_dir)
    logging.info('Done.')

  train_cost_list = list()
  while global_step_value < max_iters:
    train_start_time = time.time()
    _, desc_loss_value, attn_loss_value, reconstruction_loss_value = tfs.run(
            [train_op, desc_loss, attn_loss, reconstruction_loss])
    train_cost_list.append(time.time() - train_start_time)
    summary_writer.add_summary(merged_summary, global_step_value)

    # logging train cost
    if ((global_step_value + 1) % FLAGS.report_interval) == 0:
      start_step = global_step_value - len(train_cost_list) + 1
      logging.info("Train global steps {}-{}:cost={}\tdesc_loss={}\tattn_loss={}\treconstruction_loss={}".format(start_step, global_step_value, sum(train_cost_list), desc_loss_value, attn_loss_value, reconstruction_loss_value))
      train_cost_list.clear()
      # logging.info("train in step {}:global_step={}".format(
      #     global_step_value, tfs.run(global_step)))

    # validate once in {eval_interval*n, n \in N} steps.
    if ((global_step_value + 1) % FLAGS.eval_interval) == 0:
      logging.info("Eval start in global step:{}".format(global_step_value))
      eval_start_time = time.time()
      val_desc_loss_avg, val_desc_acc_avg, val_attn_loss_avg, val_attn_acc_avg = 0.0, 0.0, 0.0, 0.0 
      for i in range(num_eval_batches):
        val_desc_loss_, val_desc_acc_, val_attn_loss_, val_attn_acc_ = tfs.run(
                [val_desc_loss, val_desc_acc, val_attn_loss, val_attn_acc])
        val_desc_loss_avg += np.mean(val_desc_loss_)
        val_desc_acc_avg += np.mean(val_desc_acc_)
        val_attn_loss_avg += np.mean(val_attn_loss_)
        val_attn_acc_avg += np.mean(val_attn_acc_)
      val_desc_loss_avg /= num_eval_batches
      val_desc_acc_avg /= num_eval_batches
      val_attn_loss_avg /= num_eval_batches
      val_attn_acc_avg /= num_eval_batches
      eval_cost = time.time() - eval_start_time
      logging.info("Eval end in global step:{}, cost={}, desc_loss={}, desc_acc={}, attn_loss={}, attn_acc={}".format(
          global_step_value, eval_cost, val_desc_loss_avg, val_desc_acc_avg, val_attn_loss_avg, val_attn_acc_avg))
      test_summary_writer.add_summary(
          tf.Summary(value=[tf.Summary.Value(tag='eval/desc_loss', simple_value=val_desc_loss_avg),
                            tf.Summary.Value(tag='eval/desc_acc', simple_value=val_desc_acc_avg),
                            tf.Summary.Value(tag='eval/attn_loss', simple_value=val_attn_loss_avg),
                            tf.Summary.Value(tag='eval/attn_acc', simple_value=val_attn_acc_avg)]), global_step_value)
      test_summary_writer.flush()

    # save model in checkpoints
    if ((global_step_value + 1) % FLAGS.save_interval) == 0 or (global_step_value + 1) >= FLAGS.max_iters:
      save_path = saver.save(tfs, model_path, global_step=global_step_value)
      logging.info("Saved Model in file:{}".format(save_path))

    global_step_value += 1
  test_summary_writer.close()
  tfs.close()


if __name__ == '__main__':
  # npu_keras_sess = set_keras_session_npu_config()
  app.run(main)
  # close_session(npu_keras_sess)

