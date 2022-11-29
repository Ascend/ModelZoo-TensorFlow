# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""TabNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

def sparsemax(logits, name=None):
  """Computes sparsemax activations [1].
  For each batch `i` and class `j` we have
    $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$
  [1]: https://arxiv.org/abs/1602.02068
  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `logits`.
  """

  with ops.name_scope(name, "sparsemax", [logits]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    obs = array_ops.shape(logits)[0]
    dims = array_ops.shape(logits)[1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    z = logits

    # sort z
    z_sorted, _ = nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = math_ops.cumsum(z_sorted, axis=1)
    k = math_ops.range(
        1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = math_ops.maximum(k_z, 1)
    indices = array_ops.stack([math_ops.range(0, obs), k_z_safe - 1], axis=1)
    tau_sum = array_ops.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

    # calculate p
    p = math_ops.maximum(
        math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = array_ops.where(
        math_ops.logical_or(
            math_ops.equal(k_z, 0), math_ops.is_nan(z_cumsum[:, -1])),
        array_ops.fill([obs, dims], math_ops.cast(0, logits.dtype)),
        p)

    return p_safe

def glu(act, n_units):
  """Generalized linear unit nonlinear activation."""
  return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class TabNet(object):
  """TabNet model class."""

  def __init__(self,
               columns,
               num_features,
               feature_dim,
               output_dim,
               num_decision_steps,
               relaxation_factor,
               batch_momentum,
               virtual_batch_size,
               num_classes,
               epsilon=0.00001):
    """Initializes a TabNet instance.

    Args:
      columns: The Tensorflow column names for the dataset.
      num_features: The number of input features (i.e the number of columns for
        tabular data assuming each feature is represented with 1 dimension).
      feature_dim: Dimensionality of the hidden representation in feature
        transformation block. Each layer first maps the representation to a
        2*feature_dim-dimensional output and half of it is used to determine the
        nonlinearity of the GLU activation where the other half is used as an
        input to GLU, and eventually feature_dim-dimensional output is
        transferred to the next layer.
      output_dim: Dimensionality of the outputs of each decision step, which is
        later mapped to the final classification or regression output.
      num_decision_steps: Number of sequential decision steps.
      relaxation_factor: Relaxation factor that promotes the reuse of each
        feature at different decision steps. When it is 1, a feature is enforced
        to be used only at one decision step and as it increases, more
        flexibility is provided to use a feature at multiple decision steps.
      batch_momentum: Momentum in ghost batch normalization.
      virtual_batch_size: Virtual batch size in ghost batch normalization. The
        overall batch size should be an integer multiple of virtual_batch_size.
      num_classes: Number of output classes.
      epsilon: A small number for numerical stability of the entropy calcations.

    Returns:
      A TabNet instance.
    """

    self.columns = columns
    self.num_features = num_features
    self.feature_dim = feature_dim
    self.output_dim = output_dim
    self.num_decision_steps = num_decision_steps
    self.relaxation_factor = relaxation_factor
    self.batch_momentum = batch_momentum
    self.virtual_batch_size = virtual_batch_size
    self.num_classes = num_classes
    self.epsilon = epsilon

  def encoder(self, data, reuse, is_training):
    """TabNet encoder model."""

    with tf.variable_scope("Encoder", reuse=reuse):

      # Reads and normalizes input features.
      features = tf.feature_column.input_layer(data, self.columns)
      features = tf.layers.batch_normalization(
          features, training=is_training, momentum=self.batch_momentum)
      batch_size = tf.shape(features)[0]

      # Initializes decision-step dependent variables.
      output_aggregated = tf.zeros([batch_size, self.output_dim])
      masked_features = features
      mask_values = tf.zeros([batch_size, self.num_features])
      aggregated_mask_values = tf.zeros([batch_size, self.num_features])
      complemantary_aggregated_mask_values = tf.ones(
          [batch_size, self.num_features])
      total_entropy = 0

      if is_training:
        v_b = self.virtual_batch_size
      else:
        v_b = 1

      for ni in range(self.num_decision_steps):

        # Feature transformer with two shared and two decision step dependent
        # blocks is used below.

        reuse_flag = (ni > 0)

        transform_f1 = tf.layers.dense(
            masked_features,
            self.feature_dim * 2,
            name="Transform_f1",
            reuse=reuse_flag,
            use_bias=False)
        transform_f1 = tf.layers.batch_normalization(
            transform_f1,
            training=is_training,
            momentum=self.batch_momentum,
            virtual_batch_size=v_b)
        transform_f1 = glu(transform_f1, self.feature_dim)

        transform_f2 = tf.layers.dense(
            transform_f1,
            self.feature_dim * 2,
            name="Transform_f2",
            reuse=reuse_flag,
            use_bias=False)
        transform_f2 = tf.layers.batch_normalization(
            transform_f2,
            training=is_training,
            momentum=self.batch_momentum,
            virtual_batch_size=v_b)
        transform_f2 = (glu(transform_f2, self.feature_dim) +
                        transform_f1) * np.sqrt(0.5)

        transform_f3 = tf.layers.dense(
            transform_f2,
            self.feature_dim * 2,
            name="Transform_f3" + str(ni),
            use_bias=False)
        transform_f3 = tf.layers.batch_normalization(
            transform_f3,
            training=is_training,
            momentum=self.batch_momentum,
            virtual_batch_size=v_b)
        transform_f3 = (glu(transform_f3, self.feature_dim) +
                        transform_f2) * np.sqrt(0.5)

        transform_f4 = tf.layers.dense(
            transform_f3,
            self.feature_dim * 2,
            name="Transform_f4" + str(ni),
            use_bias=False)
        transform_f4 = tf.layers.batch_normalization(
            transform_f4,
            training=is_training,
            momentum=self.batch_momentum,
            virtual_batch_size=v_b)
        transform_f4 = (glu(transform_f4, self.feature_dim) +
                        transform_f3) * np.sqrt(0.5)

        if ni > 0:

          decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

          # Decision aggregation.
          output_aggregated += decision_out

          # Aggregated masks are used for visualization of the
          # feature importance attributes.
          scale_agg = tf.reduce_sum(
              decision_out, axis=1, keep_dims=True) / (
                  self.num_decision_steps - 1)
          aggregated_mask_values += mask_values * scale_agg

        features_for_coef = (transform_f4[:, self.output_dim:])

        if ni < self.num_decision_steps - 1:

          # Determines the feature masks via linear and nonlinear
          # transformations, taking into account of aggregated feature use.
          mask_values = tf.layers.dense(
              features_for_coef,
              self.num_features,
              name="Transform_coef" + str(ni),
              use_bias=False)
          mask_values = tf.layers.batch_normalization(
              mask_values,
              training=is_training,
              momentum=self.batch_momentum,
              virtual_batch_size=v_b)
          mask_values *= complemantary_aggregated_mask_values
          # mask_values = tf.contrib.sparsemax.sparsemax(mask_values)
          mask_values = sparsemax(mask_values)

          # Relaxation factor controls the amount of reuse of features between
          # different decision blocks and updated with the values of
          # coefficients.
          complemantary_aggregated_mask_values *= (
              self.relaxation_factor - mask_values)

          # Entropy is used to penalize the amount of sparsity in feature
          # selection.
          total_entropy += tf.reduce_mean(
              tf.reduce_sum(
                  -mask_values * tf.log(mask_values + self.epsilon),
                  axis=1)) / (
                      self.num_decision_steps - 1)

          # Feature selection.
          masked_features = tf.multiply(mask_values, features)

          # Visualization of the feature selection mask at decision step ni
          tf.summary.image(
              "Mask for step" + str(ni),
              tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
              max_outputs=1)

      # Visualization of the aggregated feature importances
      tf.summary.image(
          "Aggregated mask",
          tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
          max_outputs=1)

      return output_aggregated, total_entropy

  def classify(self, activations, reuse):
    """TabNet classify block."""

    with tf.variable_scope("Classify", reuse=reuse):
      logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
      predictions = tf.nn.softmax(logits)
      return logits, predictions

  def regress(self, activations, reuse):
    """TabNet regress block."""

    with tf.variable_scope("Regress", reuse=reuse):
      predictions = tf.layers.dense(activations, 1)
      return predictions
