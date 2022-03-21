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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=protected-access
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-long-lambda

r"""Docs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from npu_bridge.npu_init import *

import collections
import heapq
import os
import sys
import time
import traceback

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import common_utils
import data_utils


MODEL_SCOPE = 'model'


def eval_step_fn(params, model, images, labels):
  """Build `step_fn` for eval."""
  with tf.variable_scope(MODEL_SCOPE, reuse=True):
    student_logits = model(images, training=False)
    student_prefs = tf.nn.softmax(student_logits, axis=-1)
    student_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=student_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
    student_loss = tf.reduce_sum(student_loss) / float(params.eval_batch_size)

  with tf.variable_scope('ema', reuse=True):
    with tf.variable_scope(MODEL_SCOPE):
      ema_logits = model(images, training=False)
      ema_prefs = tf.nn.softmax(ema_logits, axis=-1)
      ema_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=ema_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
      ema_loss = tf.reduce_sum(ema_loss) / float(params.eval_batch_size)

  with tf.variable_scope('teacher', reuse=True):
    teacher_logits = model(images, training=False)
    teacher_prefs = tf.nn.softmax(teacher_logits, axis=-1)
    teacher_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=teacher_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
    teacher_loss = tf.reduce_sum(teacher_loss) / float(params.eval_batch_size)

  return student_logits, ema_logits, teacher_logits, student_loss, ema_loss, teacher_loss


def eval_ema_step_fn(params, model, images):
  with tf.variable_scope('ema', reuse=True):
    with tf.variable_scope(MODEL_SCOPE):
      ema_logits = model(images, training=False)
  return ema_logits


def eval_step_fn_with_masks(params, model, images, labels):
  """Build `step_fn` for eval."""
  with tf.variable_scope('teacher', reuse=True):
    teacher_logits = model(images, training=False)
    # teacher_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=teacher_logits, reduction=tf.losses.Reduction.SUM)
    teacher_labels = tf.nn.softmax(teacher_logits / params.uda_temp, axis=-1)
    teacher_max_probs = tf.reduce_max(teacher_labels, axis=-1, keepdims=True)
    teacher_masks = tf.greater_equal(teacher_max_probs, 0.13)
    teacher_masks = tf.cast(teacher_masks, tf.float32)
    teacher_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=teacher_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
    teacher_loss = tf.reduce_sum(teacher_loss) / float(params.eval_batch_size)

  with tf.variable_scope(MODEL_SCOPE, reuse=True):
    student_logits = model(images, training=False)
    # student_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=student_logits, reduction=tf.losses.Reduction.SUM)
    student_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=student_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
    student_loss = tf.reduce_sum(student_loss) / float(params.eval_batch_size)
    student_labels = tf.nn.softmax(student_logits / params.uda_temp, axis=-1)
    student_max_probs = tf.reduce_max(student_labels, axis=-1, keepdims=True)
    student_masks = tf.greater_equal(student_max_probs, 0.13)
    student_masks = tf.cast(student_masks, tf.float32)

  with tf.variable_scope('ema', reuse=True):
    with tf.variable_scope(MODEL_SCOPE):
      ema_logits = model(images, training=False)
      # ema_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=ema_logits, reduction=tf.losses.Reduction.SUM)
      ema_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=ema_logits, label_smoothing=params.label_smoothing, reduction=tf.losses.Reduction.NONE)
      ema_loss = tf.reduce_sum(ema_loss) / float(params.eval_batch_size)
      ema_labels = tf.nn.softmax(ema_logits / params.uda_temp, axis=-1)
      ema_max_probs = tf.reduce_max(ema_labels, axis=-1, keepdims=True)
      ema_masks = tf.greater_equal(ema_max_probs, 0.13)
      ema_masks = tf.cast(ema_masks, tf.float32)
  return teacher_logits, student_logits, ema_logits, teacher_loss, student_loss, ema_loss, teacher_masks, student_masks, ema_masks 


def eval_accuracy(labels, logits):
  sorted_indices = np.argsort(logits, axis=-1)
  def _top_k(k):
    in_top_k = np.any(sorted_indices[:, -k:] == np.argmax(labels, axis=1).reshape(-1, 1), axis=-1)
    total = np.sum(in_top_k.astype(np.float32))
    return total
  top_1, top_5 = _top_k(k=1), _top_k(k=5)
  return top_1, top_5


def eval_accuracy_with_mask(labels, logits, masks):
  total_size = np.sum(masks)
  sorted_indices = np.argsort(logits, axis=-1)
  def _top_k(k):
    in_top_k = np.any(sorted_indices[:, -k:] == np.argmax(labels, axis=1).reshape(-1, 1), axis=-1)
    total = np.sum(np.multiply(in_top_k.astype(np.float32).reshape(-1, 1), masks))
    return total
  top_1, top_5 = _top_k(k=1), _top_k(k=5)
  return top_1, top_5, total_size


class UDA(object):
  """UDA (https://arxiv.org/abs/1904.12848)."""

  def __init__(self):
    self.step_info = collections.OrderedDict()

  def outfeed_signature(self):
    """Returns the sigature of `step_info` as returned by `step_fn`."""
    return self.step_info

  @staticmethod
  def build_uda_cross_entropy(params, model, all_images, l_labels):
    """Compute the UDA loss."""
    train_batch_size = params.train_batch_size
    num_replicas = 1
    uda_data = params.uda_data
    batch_size = train_batch_size // num_replicas

    labels = {}
    if l_labels.dtype == tf.int32:  # l_labels is sparse. turn into one_hot
      labels['l'] = tf.one_hot(l_labels, params.num_classes, dtype=tf.float32)
    else:
      labels['l'] = l_labels

    global_step = tf.train.get_or_create_global_step()

    masks = {}
    logits = {}
    cross_entropy = {}
    all_logits = model(all_images, training=True)

    logits['l'], logits['u_ori'], logits['u_aug'] = tf.split(
        all_logits, [batch_size, batch_size*uda_data, batch_size*uda_data], 0)

    # sup loss
    cross_entropy['l'] = tf.losses.softmax_cross_entropy(
        onehot_labels=labels['l'],
        logits=logits['l'],
        label_smoothing=params.label_smoothing,
        reduction=tf.losses.Reduction.NONE)
    probs = tf.nn.softmax(logits['l'], axis=-1)
    correct_probs = tf.reduce_sum(labels['l']*probs, axis=-1)
    r = tf.cast(global_step, tf.float32) / float(params.num_train_steps)
    l_threshold = r * (1. - 1./params.num_classes) + 1. / params.num_classes
    masks['l'] = tf.less_equal(correct_probs, l_threshold)
    masks['l'] = tf.cast(masks['l'], tf.float32)
    masks['l'] = tf.stop_gradient(masks['l'])
    cross_entropy['l'] = tf.reduce_sum(cross_entropy['l']) / float(
        train_batch_size)

    # unsup loss
    labels['u_ori'] = tf.nn.softmax(logits['u_ori'] / params.uda_temp, axis=-1)
    labels['u_ori'] = tf.stop_gradient(labels['u_ori'])

    cross_entropy['u'] = (labels['u_ori'] *
                          tf.nn.log_softmax(logits['u_aug'], axis=-1))
    largest_probs = tf.reduce_max(labels['u_ori'], axis=-1, keepdims=True)
    masks['u'] = tf.greater_equal(largest_probs, params.uda_threshold)
    masks['u'] = tf.cast(masks['u'], tf.float32)
    masks['u'] = tf.stop_gradient(masks['u'])
    cross_entropy['u'] = tf.reduce_sum(-cross_entropy['u']*masks['u']) / float(train_batch_size*uda_data)
    return logits, labels, masks, cross_entropy

  def step_fn(self, params, model):
    """Separate implementation."""
    train_batch_size = params.train_batch_size
    num_replicas = 1
    batch_size = train_batch_size // 1

    dtypes = [
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32,
        tf.bfloat16 if params.use_bfloat16 else tf.float32]
    shapes = [
        [batch_size, params.image_size, params.image_size, 3],
        [batch_size, params.num_classes],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3],
        [batch_size*params.uda_data, params.image_size, params.image_size, 3]]

    with tf.device('/cpu:0'):
        (l_images, l_labels, u_images_ori,
         u_images_aug) = tf.raw_ops.InfeedDequeueTuple(dtypes=dtypes,
                                                       shapes=shapes)

    all_images = tf.concat([l_images, u_images_ori, u_images_aug], axis=0)
    global_step = tf.train.get_or_create_global_step()
    num_replicas = tf.cast(params.num_replicas, tf.float32)

    with tf.variable_scope(MODEL_SCOPE, reuse=tf.AUTO_REUSE):
      _, _, masks, cross_entropy = UDA.build_uda_cross_entropy(
          params, model, all_images, l_labels)

    l2_reg_rate = tf.cast(params.weight_decay / params.num_replicas, tf.float32)
    weight_dec = common_utils.get_l2_loss()
    uda_weight = params.uda_weight * tf.minimum(
        1., tf.cast(global_step, tf.float32) / float(params.uda_steps))
    total_loss = (cross_entropy['u'] * uda_weight +
                  cross_entropy['l'] +
                  weight_dec * l2_reg_rate)
    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
    gradients = [tf.tpu.cross_replica_sum(g) for g in gradients]
    gradients, grad_norm = tf.clip_by_global_norm(gradients, params.grad_bound)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    learning_rate, optimizer = common_utils.get_optimizer(params)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(zip(gradients, variables),
                                           global_step=global_step)

    with tf.control_dependencies([train_op]):
      ema_train_op = common_utils.setup_ema(
          params, f'{MODEL_SCOPE}/{model.name}')

    with tf.control_dependencies([ema_train_op]):
      logs = collections.OrderedDict()
      logs['global_step'] = tf.cast(global_step, tf.float32)
      logs['loss/total'] = total_loss
      logs['loss/cross_entropy'] = cross_entropy['l']
      logs['loss/lr'] = tf.identity(learning_rate) / num_replicas
      logs['loss/grad_norm'] = tf.identity(grad_norm) / num_replicas
      logs['loss/weight_dec'] = weight_dec / num_replicas

      logs['uda/cross_entropy'] = cross_entropy['u']
      logs['uda/u_ratio'] = tf.reduce_mean(masks['u']) / num_replicas
      logs['uda/l_ratio'] = tf.reduce_mean(masks['l']) / num_replicas
      logs['uda/weight'] = uda_weight / num_replicas

      tensors = [tf.expand_dims(t, axis=0) for t in logs.values()]
      self.step_info = {k: [tf.float32, [1]] for k in logs.keys()}
      outfeed_enqueue_op = tf.cond(
          common_utils.should_log(params),
          lambda: tf.raw_ops.OutfeedEnqueueTuple(inputs=tensors), tf.no_op)
    return outfeed_enqueue_op


class MPL(object):
    """Meta Pseudo Labels."""

    def __init__(self):
        self.step_info = collections.OrderedDict()

    def outfeed_signature(self):
        """Returns the sigature of `step_info` as returned by `step_fn`."""
        return self.step_info

    def step_fn(self, params, model, l_images, l_labels, u_images_ori, u_images_aug):
        """Separate implementation."""
        train_batch_size = params.train_batch_size
        uda_data = params.uda_data
        batch_size = train_batch_size

        shapes = [
            [batch_size, params.image_size, params.image_size, 3],
            [batch_size, params.num_classes],
            [batch_size*params.uda_data, params.image_size, params.image_size, 3],
            [batch_size*params.uda_data, params.image_size, params.image_size, 3]]

        global_step = tf.train.get_or_create_global_step()
        all_images = tf.concat([l_images, u_images_ori, u_images_aug], axis=0)

        # all calls to teacher
        with tf.variable_scope('teacher', reuse=tf.AUTO_REUSE):
            logits, labels, masks, cross_entropy = UDA.build_uda_cross_entropy(params, model, all_images, l_labels)
            # teacher_acc, teacher_acc_op = tf.metrics.accuracy(labels=tf.argmax(labels['l'], 1), predictions=tf.argmax(logits['l'], 1))
            # teacher_equal = (tf.argmax(labels['l'], 1) == tf.argmax(logits['l'], 1))
            # teacher_equal = tf.cast(teacher_equal, tf.float32)
            # teacher_acc = tf.reduce_mean(teacher_equal)

        # 1st call to student
        with tf.variable_scope(MODEL_SCOPE):
            u_aug_and_l_images = tf.concat([u_images_aug, l_images], axis=0)
            logits['s_on_u_aug_and_l'] = model(u_aug_and_l_images, training=True)
            logits['s_on_u'], logits['s_on_l_old'] = tf.split(
                  logits['s_on_u_aug_and_l'],
                  [tf.shape(u_images_aug)[0], tf.shape(l_images)[0]], axis=0)

        # for backprop
        cross_entropy['s_on_u'] = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], -1)),
            logits=logits['s_on_u'],
            label_smoothing=params.label_smoothing,
            reduction=tf.losses.Reduction.NONE)
        cross_entropy['s_on_u'] = tf.reduce_sum(cross_entropy['s_on_u']) / float(
            train_batch_size*uda_data)

        # for Taylor
        cross_entropy['s_on_l_old'] = tf.losses.softmax_cross_entropy(
            onehot_labels=labels['l'],
            logits=logits['s_on_l_old'],
            reduction=tf.losses.Reduction.NONE)
        cross_entropy['s_on_l_old'] = tf.reduce_sum(cross_entropy['s_on_l_old']) / float(train_batch_size)
        shadow = tf.get_variable(name='cross_entropy_old', shape=[], trainable=False, dtype=tf.float32)
        shadow_update = tf.assign(shadow, cross_entropy['s_on_l_old'])

        w_s = {}
        g_s = {}
        g_n = {}
        lr = {}
        optim = {}
        w_s['s'] = [w for w in tf.trainable_variables() if w.name.lower().startswith(MODEL_SCOPE)]
        g_s['s_on_u'] = tf.gradients(cross_entropy['s_on_u'], w_s['s'])
        logging.info("student trainable variables={}".format(len(w_s['s'])))

        lr['s'] = common_utils.get_learning_rate(
            params,
            initial_lr=params.mpl_student_lr,
            num_warmup_steps=params.mpl_student_lr_warmup_steps,
            num_wait_steps=params.mpl_student_lr_wait_steps)
        lr['s'], optim['s'] = common_utils.get_optimizer(params, learning_rate=lr['s'])
        optim['s']._create_slots(w_s['s'])
        ### significant bug: update_ops is empty
        # update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #               if op.name.startswith(f'train/{MODEL_SCOPE}/')]
        student_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
                              if op.name.startswith('model/')]  # batch_norm update_ops of student model
        # debug
        all_ops = [op.name for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]
        logging.info("all_ops={}".format(len(all_ops)))

        with tf.control_dependencies(student_update_ops + [shadow_update]):
            g_s['s_on_u'] = common_utils.add_weight_decay(params, w_s['s'], g_s['s_on_u'])
            g_s['s_on_u'], g_n['s_on_u'] = tf.clip_by_global_norm(g_s['s_on_u'], params.grad_bound)
            train_op = optim['s'].apply_gradients(list(zip(g_s['s_on_u'], w_s['s'])))

            with tf.control_dependencies([train_op]):
                ema_train_op = common_utils.setup_ema(params, name_scope=f'{MODEL_SCOPE}/{model.name}')

        # 2nd call to student
        with tf.control_dependencies([ema_train_op]):
            with tf.variable_scope(MODEL_SCOPE, reuse=tf.AUTO_REUSE):
                logits['s_on_l_new'] = model(l_images, training=True)
        cross_entropy['s_on_l_new'] = tf.losses.softmax_cross_entropy(
            onehot_labels=labels['l'],
            logits=logits['s_on_l_new'],
            reduction=tf.losses.Reduction.NONE)
        cross_entropy['s_on_l_new'] = tf.reduce_sum(cross_entropy['s_on_l_new']) / float(train_batch_size)
        
        dot_product = cross_entropy['s_on_l_new'] - shadow
        moving_dot_product = tf.get_variable('moving_dot_product', shape=[], trainable=False, dtype=tf.float32)
        moving_dot_product_update = tf.assign_sub(moving_dot_product, 0.01 * (moving_dot_product - dot_product))
        with tf.control_dependencies([moving_dot_product_update]):
            dot_product = dot_product - moving_dot_product
            dot_product = tf.stop_gradient(dot_product)
        cross_entropy['mpl'] = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.stop_gradient(tf.nn.softmax(logits['u_aug'], axis=-1)),
            logits=logits['u_aug'],
            reduction=tf.losses.Reduction.NONE)
        cross_entropy['mpl'] = tf.reduce_sum(cross_entropy['mpl']) / float(train_batch_size*uda_data)

        # teacher train op
        #dot_product2 = shadow - cross_entropy['s_on_l_new']
        #dot_product2 = tf.stop_gradient(dot_product2)

        ### TODO: add l2 regularization
        #l2_reg_rate = tf.cast(params.weight_decay, tf.float32)
        #weight_dec = common_utils.get_l2_loss()

        uda_weight = params.uda_weight * tf.minimum(1., tf.cast(global_step, tf.float32) / float(params.uda_steps))
        teacher_loss = (cross_entropy['u'] * uda_weight + 
                        cross_entropy['l'] + 
                        cross_entropy['mpl'] * dot_product)
        w_s['t'] = [w for w in tf.trainable_variables() if 'teacher' in w.name]
        g_s['t'] = tf.gradients(teacher_loss, w_s['t'])
        logging.info("teacher trainable variables={}".format(len(w_s['t'])))

        g_s['t'] = common_utils.add_weight_decay(params, w_s['t'], g_s['t'])
        g_s['t'], g_n['t'] = tf.clip_by_global_norm(g_s['t'], params.grad_bound)
        ### TODO: learning rate
        lr['t'] = common_utils.get_learning_rate(
            params,
            initial_lr=params.mpl_teacher_lr,
            num_warmup_steps=params.mpl_teacher_lr_warmup_steps,
            num_wait_steps=0)
        lr['t'], optim['t'] = common_utils.get_optimizer(params, learning_rate=lr['t'])

        # added: batch_norm update_ops of student model(second round) and teacher model
        all_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
                          if not op.name.startswith('model/')]
        with tf.control_dependencies(all_update_ops):
            teacher_train_op = optim['t'].apply_gradients(list(zip(g_s['t'], w_s['t'])), global_step=global_step)

        with tf.control_dependencies([teacher_train_op]):
            cross_entropy_student_on_u = cross_entropy['s_on_u']
            cross_entropy_student_on_l = cross_entropy['s_on_l_new']
            cross_entropy_teacher_on_u = cross_entropy['u']
            cross_entropy_teacher_on_l = cross_entropy['l']
            lr_student = tf.identity(lr['s'])
            lr_teacher = tf.identity(lr['t'])
            mpl_dot_product = dot_product
            mpl_moving_dot_product = moving_dot_product
            uda_u_ratio = tf.reduce_mean(masks['u'])
            uda_l_ratio = tf.reduce_mean(masks['l'])
            mpl_uda_weight = uda_weight
            mpl_teacher_loss = teacher_loss
        tf.summary.scalar("cross_entropy_student_on_u", cross_entropy_student_on_u)
        tf.summary.scalar("cross_entropy_student_on_l", cross_entropy_student_on_l)
        tf.summary.scalar("cross_entropy_teacher_on_u", cross_entropy_teacher_on_u)
        tf.summary.scalar("cross_entropy_teacher_on_l", cross_entropy_teacher_on_l)
        tf.summary.scalar("lr_student", lr_student)
        tf.summary.scalar("lr_teacher", lr_teacher)
        tf.summary.scalar("mpl_dot_product", mpl_dot_product)
        tf.summary.scalar("mpl_moving_dot_product", mpl_moving_dot_product)
        tf.summary.scalar("uda_u_ratio", uda_u_ratio)
        tf.summary.scalar("uda_l_ratio", uda_l_ratio)
        tf.summary.scalar("uda_weight", uda_weight)
        tf.summary.scalar("mpl_teacher_loss", mpl_teacher_loss)
        merged_summary_op = tf.summary.merge_all()

        return merged_summary_op, teacher_train_op, logits['l'], labels['l']
