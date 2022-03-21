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
"""Basic Model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf


def update_argparser(parser):
  parser.add_argument(
      '--learning_rate',
      help='Learning rate',
      default=0.001,
  )


def model_fn(features, labels, mode, params, config):
  predictions = None
  loss = None
  train_op = None
  eval_metric_ops = None
  export_outputs = None

  x = features['feature']
  x = tf.layers.dense(x, 10, activation=tf.nn.relu)
  x = tf.layers.dense(x, 1, activation=None)
  predictions = x

  if mode == tf.estimator.ModeKeys.PREDICT:
    export_outputs = {
        tf.saved_model.signature_constants.REGRESS_METHOD_NAME:
            tf.estimator.export.RegressionOutput(predictions)
    }
  else:
    labels = labels['label']
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'MSE':
              tf.metrics.mean_squared_error(
                  labels=labels, predictions=predictions),
      }
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = npu_tf_optimizer(tf.train.AdamOptimizer(params.learning_rate)).minimize(
          loss,
          global_step=tf.train.get_or_create_global_step(),
      )

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=export_outputs,
  )
