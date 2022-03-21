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

from npu_bridge.npu_init import *
import os
import re
import tensorflow as tf

from src import Detector, AnchorGenerator, FeatureExtractor, Evaluator



def model_fn(features, labels, mode, params, config):
    """This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """
    # the base network
    print("++++++++++++++++++++++model_start")
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    feature_extractor = FeatureExtractor(is_training)

    # anchor maker
    anchor_generator = AnchorGenerator()

    # add box/label predictors to the feature extractor
    detector = Detector(features['images'], feature_extractor, anchor_generator)

    # add NMS to the graph
    if not is_training:
        predictions = detector.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes=params['max_boxes']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()
    # create localization and classification losses
    losses = detector.loss(labels, params)
    tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
    tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
    # tf.summary.scalar('regularization_loss', regularization_loss)
    # tf.summary.scalar('localization_loss', losses['localization_loss'])
    # tf.summary.scalar('classification_loss', losses['classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        filenames = features['filenames']
        batch_size = filenames.shape.as_list()[0]
        assert batch_size == 1

        with tf.name_scope('evaluator'):
            evaluator = Evaluator()
            eval_metric_ops = evaluator.get_metric_ops(filenames, labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        # tf.summary.scalar('learning_rate', learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = npu_distributed_optimizer_wrapper(tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True))
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    # for g, v in grads_and_vars:
    #     tf.summary.histogram(v.name[:-2] + '_hist', v)
    #     tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
    # print("++++++++++++++++++++++model_end")
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)

