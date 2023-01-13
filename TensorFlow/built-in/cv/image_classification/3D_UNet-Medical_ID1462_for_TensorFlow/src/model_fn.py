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

"""
3D U-Net network, for prostate MRI scans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf
import time

from src.network import unet_3d_network


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self):
        self.current_step = 0
        self.t0 = None

    def before_run(self, run_context):
        self.t0 = time.time()

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        if self.current_step <= 5:
            tf.logging.info(f"################ ... current_step = {self.current_step}, batch_time = {batch_time}")
        self.current_step += 1

def model_fn(features, labels, mode, params):
    """
    Custom estimator setup as per docs and guide:
    https://www.tensorflow.org/guide/custom_estimators

    Several ideas taken from: https://github.com/cs230-stanford/
    cs230-code-examples/tree/master/tensorflow/vision

    Args:
        features: This is batch_features from input_fn.
        labels: This is batch_labels from input_fn.
        mode (:class:`tf.estimator.ModeKeys`): Train, eval, or predict.
        params (dict): Params for setting up the model. Expected keys are:
            depth (int): Depth of the architecture.
            n_base_filters (int): Number of conv3d filters in the first layer.
            num_classes (int): Number of mutually exclusive output classes.
            class_weights (:class:`numpy.array`): Weight of each class to use.
            learning_rate (float): LR to use with Adam.
            batch_norm (bool): Whether to use batch_norm in the conv3d blocks.
            display_steps (int): How often to log about progress.

    Returns:
        :class:`tf.estimator.Estimator`: A 3D U-Net network, as TF Estimator.
    """

    # -------------------------------------------------------------------------
    # get logits from 3D U-Net
    # -------------------------------------------------------------------------

    training = mode == tf.estimator.ModeKeys.TRAIN
    logits = unet_3d_network(inputs=features['x'], params=params, training=training)

    # -------------------------------------------------------------------------
    # predictions - for PREDICT and EVAL modes
    # -------------------------------------------------------------------------

    prediction = tf.argmax(logits, axis=-1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': prediction,
            'truth': tf.argmax(features['y'], -1),
            'probabilities': tf.nn.softmax(logits, axis=-1)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # -------------------------------------------------------------------------
    # loss - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------

    # weighted softmax, see https://stackoverflow.com/a/44563055
    class_weights = tf.cast(tf.constant(params['class_weights']), tf.float32)
    class_weights = tf.reduce_sum(
        tf.cast(features['x'], tf.float32) * class_weights, axis=-1
    )
    loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        weights=class_weights
    )
    tf.summary.scalar('loss', loss)

    # -------------------------------------------------------------------------
    # metrics: mean IOU - for TRAIN and EVAL modes
    # -------------------------------------------------------------------------
    '''
    labels_dense = tf.argmax(labels, -1)
    iou = tf.metrics.mean_iou(
        labels=labels_dense,
        predictions=tf.cast(prediction, tf.int32),
        num_classes=params['num_classes'],
        name='iou_op'
    )
    metrics = {'iou': iou}
    tf.summary.scalar('iou', iou[0])
    '''
    if mode == tf.estimator.ModeKeys.EVAL:
        labels_dense = tf.argmax(labels, -1)
        iou = tf.metrics.mean_iou(
            labels=labels_dense,
            predictions=tf.cast(prediction, tf.int32),
            num_classes=params['num_classes'],
            name='iou_op'
        )
        metrics = {'iou': iou}
        tf.summary.scalar('iou', iou[0])
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # -------------------------------------------------------------------------
    # train op - for TRAIN
    # -------------------------------------------------------------------------

    assert mode == tf.estimator.ModeKeys.TRAIN

    # modify for NPU start
    # optimizer = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=params['learning_rate']))

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    if 'precision_mode' in params:
        if params['precision_mode'] == "allow_mix_precision":
            print("precision_mode=====================>", params['precision_mode'])
    loss_scale_manager = ExponentialUpdateLossScaleManager(
        init_loss_scale=2**32,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=2,
        decr_ratio=0.5)
    optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
    # modify for NPU end

    global_step = tf.train.get_or_create_global_step()

    if params['batch_norm']:
        # as per TF batch_norm docs and also following https://goo.gl/1UVeYK
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[BenchmarkLoggingHook()])
