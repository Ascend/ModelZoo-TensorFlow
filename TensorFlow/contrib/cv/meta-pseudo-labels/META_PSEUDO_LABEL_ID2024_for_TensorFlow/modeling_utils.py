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
# pylint: disable=invalid-unary-operand-type
# pylint: disable=g-long-lambda
# pylint: disable=g-direct-tensorflow-import

r"""Custom ops."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from npu_bridge.npu_init import *

import os
import sys

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import common_utils


USE_BFLOAT16 = False
NUM_XLA_SHARDS = -1


def floatx():
  return tf.bfloat16 if USE_BFLOAT16 else tf.float32


def use_bfloat16():
  global USE_BFLOAT16
  USE_BFLOAT16 = True


def set_xla_sharding(num_xla_shards):
  global NUM_XLA_SHARDS
  NUM_XLA_SHARDS = num_xla_shards


def _cross_replica_avg(params, x):
  """Average the value `x` across `params.num_replicas` replicas."""
  return tf.tpu.cross_replica_sum(x) / tf.cast(params.num_replicas, x.dtype)


def get_variable(name, shape, initializer, trainable=True,
                 convert_if_using_bfloat16=True):
  """Create variable and convert to `tf.bfloat16` if needed."""
  w = tf.get_variable(name=name,
                      shape=shape,
                      initializer=initializer,
                      trainable=trainable,
                      use_resource=True)
  if USE_BFLOAT16 and convert_if_using_bfloat16:
    w = tf.cast(w, tf.bfloat16)
  return w


def log_tensor(x, training):
  """Prints a tensor."""
  if training:
    logging.info(f'{x.name:<90} {x.device} {x.shape}')


def _conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels."""
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape[-4:]
  fan_out = int(kernel_height * kernel_width * out_filters)
  # return tf.random.normal(
  #     shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)
  with npu_scope.keep_dtype_scope():
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=tf.float32)


def _dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels."""
  del partition_info
  init_range = 1. / np.sqrt(shape[-1])
  return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def conv2d(x, filter_size, num_out_filters, stride=1,
           use_bias=False, padding='SAME', data_format='NHWC', name='conv2d',
           w=None, b=None):
  """Conv."""
  with tf.variable_scope(name):
    num_inp_filters = x.shape[-1].value

    w = tf.get_variable(
        name='kernel',
        shape=[filter_size, filter_size, num_inp_filters, num_out_filters],
        initializer=_conv_kernel_initializer,
        trainable=True,
        use_resource=True)

    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1],
                     padding=padding, data_format=data_format)

    if use_bias:
      if b is None:
        b = tf.get_variable(
            name='bias',
            shape=[num_out_filters],
            initializer=tf.initializers.zeros(),
            trainable=True,
            use_resource=True)
      x = tf.nn.bias_add(x, b, name='bias_add')
    return x


def dw_conv2d(x, filter_size, stride, depth_multiplier=1,
              padding='SAME', data_format='NHWC', name='dw_conv_2d', w=None):
  """Custom depthwise conv."""
  if depth_multiplier > 1:
    raise NotImplementedError('Bite me!')

  with tf.variable_scope(name):
    num_inp_filters = x.shape[-1].value
    w = tf.get_variable(
        name='depthwise_kernel',
        shape=[filter_size, filter_size, num_inp_filters, 1],
        initializer=_conv_kernel_initializer,
        trainable=True,
        use_resource=True)

    w = common_utils.shard_weight(w, NUM_XLA_SHARDS)

    if USE_BFLOAT16:
      w = tf.cast(w, tf.bfloat16)
    x = tf.nn.depthwise_conv2d(x, filter=w, strides=[1, stride, stride, 1],
                               padding=padding, data_format=data_format)
    return x


def dense(x, num_outputs, use_bias=True, name='dense'):
  """Custom fully connected layer."""
  num_inputs = x.shape[-1].value
  with tf.variable_scope(name):
    w = tf.get_variable(
        name='kernel',
        shape=[num_inputs, num_outputs],
        initializer=_dense_kernel_initializer,
        trainable=True,
        use_resource=True)
    x = tf.linalg.matmul(x, w)
    if use_bias:
      b = tf.get_variable(
          name='bias',
          shape=[num_outputs],
          initializer=tf.initializers.zeros(),
          trainable=True,
          use_resource=True)
      x = tf.nn.bias_add(x, b, name='bias_add')
    return x


def avg_pool(x, filter_size, stride, padding='SAME', name='avg_pool'):
  """Avg pool."""
  x = tf.nn.avg_pool(
      x,
      ksize=[filter_size, filter_size],
      strides=[1, stride, stride, 1],
      padding=padding,
      name=name)
  return x


def max_pool(x, filter_size, stride, padding='SAME', name='max_pool'):
  """Avg pool."""
  x = tf.nn.max_pool(
      x,
      ksize=[filter_size, filter_size],
      strides=[1, stride, stride, 1],
      padding=padding,
      name=name)
  return x


def relu(x, leaky=0.2, name='relu'):
  """Leaky ReLU."""
  return tf.nn.leaky_relu(x, alpha=leaky, name=name)


def batch_norm(x, params, training, name='batch_norm', **kwargs):
  """Wrapped `batch_norm`."""
  ## with npu_scope.keep_dtype_scope():
  return tf.layers.batch_normalization(x, training=training, name=name, epsilon=params.batch_norm_epsilon)
  #return sync_batch_norm(x, params, training, name=name, **kwargs)


def sync_batch_norm(x, params, training, name='batch_norm'):
  """Sync batch_norm."""
  size = x.shape[-1].value

  with tf.variable_scope(name):
    gamma = tf.get_variable(name='gamma',
                            shape=[size],
                            initializer=tf.initializers.ones(),
                            trainable=True)
    beta = tf.get_variable(name='beta',
                           shape=[size],
                           initializer=tf.initializers.zeros(),
                           trainable=True)
    moving_mean = tf.get_variable(name='moving_mean',
                                  shape=[size],
                                  initializer=tf.initializers.zeros(),
                                  trainable=False)
    moving_variance = tf.get_variable(name='moving_variance',
                                      shape=[size],
                                      initializer=tf.initializers.ones(),
                                      trainable=False)

  x = tf.cast(x, tf.float32)
  if training:
    mean, variance = tf.nn.moments(x, [0, 1, 2])
    x = tf.nn.batch_normalization(
        x, mean=mean, variance=variance, offset=beta, scale=gamma,
        variance_epsilon=params.batch_norm_epsilon)
    if (isinstance(moving_mean, tf.Variable) and
        isinstance(moving_variance, tf.Variable)):
      decay = tf.cast(1. - params.batch_norm_decay, tf.float32)
      def u(moving, normal, name):
        diff = decay * (moving - normal)
        return tf.assign_sub(moving, diff, use_locking=True, name=name)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                           u(moving_mean, mean, name='moving_mean'))
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                           u(moving_variance, variance, name='moving_variance'))
      return x
    else:
      return x, mean, variance
  else:
    x = tf.nn.batch_normalization(
        x, mean=moving_mean, variance=moving_variance, offset=beta,
        scale=gamma, variance_epsilon=params.batch_norm_epsilon)
    return x


def gpu_batch_norm(x, params, training=True, name='batch_norm'):
  with tf.variable_scope(name):
    x = tf.layers.batch_normalization(x, momentum=0.01, epsilon=1e-3, training=training)
  return x 


def gpu_batch_norm_ori(x, params, training=True, name='batch_norm'):
  """Async batch_norm."""
  shape = [x.shape[-1].value]
  with tf.variable_scope(name):
    gamma = get_variable('gamma', shape, tf.initializers.ones())
    beta = get_variable('beta', shape, tf.initializers.zeros())
    moving_mean = tf.get_variable(
        name='moving_mean',
        shape=shape,
        initializer=tf.initializers.zeros(),
        trainable=False,
        use_resource=True)
    moving_variance = tf.get_variable(
        name='moving_variance',
        shape=shape,
        initializer=tf.initializers.ones(),
        trainable=False,
        use_resource=True)

  if training:
    x, mean, variance = tf.nn.fused_batch_norm(
        x,
        scale=gamma,
        offset=beta,
        epsilon=params.batch_norm_epsilon,
        is_training=True,
    )

    def u(moving, normal):
      decay = tf.cast(1. - params.batch_norm_decay, tf.float32)
      return moving.assign_sub(decay * (moving - normal))

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u(moving_mean, mean))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u(moving_variance, variance))
  else:
    x, _, _ = tf.nn.fused_batch_norm(
        x,
        scale=gamma,
        offset=beta,
        mean=moving_mean,
        variance=moving_variance,
        epsilon=params.batch_norm_epsilon,
        is_training=False,
    )
  return x


def wrn_block(x, params, num_out_filters, stride, training=True,
              name='wrn_block'):
  """WideResNet block."""

  with tf.variable_scope(name):
    num_inp_filters = x.shape[-1].value
    residual = x
    with tf.variable_scope('conv_3x3_1'):
      x = batch_norm(x, params, training)  ### TODO
      ### x = gpu_batch_norm(x, params, training)
      if stride == 2 or num_inp_filters != num_out_filters:
        residual = x
      x = relu(x)  ### TODO: try slope=0.1? 
      x = conv2d(x, 3, num_out_filters, stride)

    with tf.variable_scope('conv_3x3_2'):
      x = batch_norm(x, params, training)  ### TODO
      ### x = gpu_batch_norm(x, params, training)
      x = relu(x)
      x = conv2d(x, 3, num_out_filters, 1)

    with tf.variable_scope('residual'):
      if stride == 2 or num_inp_filters != num_out_filters:
        residual = relu(residual)
        residual = conv2d(residual, 1, num_out_filters, stride)

      x = x + residual
      log_tensor(x, True)
  return x


def resnet_block(x, params, num_out_filters, stride, training=True,
                 bottleneck_rate=4, name='resnet_block'):
  """ResNet-50 bottleneck block."""

  num_bottleneck_filters = num_out_filters // bottleneck_rate
  with tf.variable_scope(name):
    residual = x
    num_inp_filters = residual.shape[-1].value
    with tf.variable_scope('conv_1x1_1'):
      x = conv2d(x, 1, num_bottleneck_filters, 1)
      x = batch_norm(x, params, training)
      x = relu(x, leaky=0.)

    with tf.variable_scope('conv_3x3'):
      x = conv2d(x, 3, num_bottleneck_filters, stride)
      x = batch_norm(x, params, training)
      x = relu(x, leaky=0.)

    with tf.variable_scope('conv_1x1_2'):
      x = conv2d(x, 1, num_out_filters, 1)
      x = batch_norm(x, params, training)

    with tf.variable_scope('residual'):
      if stride == 2 or num_inp_filters != num_out_filters:
        residual = conv2d(residual, 1, num_out_filters, stride)
        residual = batch_norm(residual, params, training)

      x = relu(x + residual, leaky=0.)
      log_tensor(x, True)
  return x


def dropout(x, drop_rate, training):
  """Dropout."""
  if training:
    # return tf.nn.dropout(x, rate=drop_rate)
    # cann_used
    return npu_ops.dropout(x, keep_prob=1-drop_rate)
  else:
    return x


@tf.custom_gradient
def swish(x, name='swish'):
  """Compute `swish(x) = x*sigmoid(x)` from https://arxiv.org/abs/1710.05941.

  Args:
    x: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """

  def grad(dy):
    """Gradient for the Swish activation function."""
    with tf.control_dependencies([dy]):
      sigmoid_x = tf.nn.sigmoid(x)
    one = tf.cast(1., x.dtype)
    activation_grad = sigmoid_x * (one + x * (one - sigmoid_x))
    return dy * activation_grad

  return tf.multiply(x, tf.nn.sigmoid(x), name=name), grad


def squeeze_and_excitation(x, num_se_filters, name='se', weights=None):
  """Squeeze and Excitation layer."""
  num_inp_filters = x.shape[-1].value
  with tf.variable_scope(name):
    se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    se = conv2d(se, 1, num_se_filters, 1, use_bias=True, name='reduce',
                w=weights['se/reduce/conv/w'] if weights is not None else None,
                b=weights['se/reduce/conv/b'] if weights is not None else None)
    se = swish(se)
    se = conv2d(se, 1, num_inp_filters, 1, use_bias=True, name='expand',
                w=weights['se/expand/conv/w'] if weights is not None else None,
                b=weights['se/expand/conv/b'] if weights is not None else None)
    se = tf.nn.sigmoid(se)
    return se * x


def mb_conv_block(x, params, filter_size, num_out_filters, stride, training,
                  stochastic_depth_drop_rate=0., expand_ratio=1, use_se=True,
                  se_ratio=0.25, name='mb_conv'):
  """Mobile-Inverse Convolutional Block."""

  num_inp_filters = x.shape[-1].value
  num_exp_filters = num_inp_filters * expand_ratio

  with tf.variable_scope(name):
    residual = x
    if expand_ratio > 1:
      with tf.variable_scope('expand'):
        x = conv2d(x, 1, num_exp_filters, 1)
        x = batch_norm(x, params, training)
        x = swish(x)

    with tf.variable_scope('depthwise'):
      x = dw_conv2d(x, filter_size, stride)
      x = batch_norm(x, params, training)
      x = swish(x)

    if use_se:
      num_se_filters = max(1, int(num_inp_filters * se_ratio))
      x = squeeze_and_excitation(x, num_se_filters)

    with tf.variable_scope('output'):
      x = conv2d(x, 1, num_out_filters, 1)
      x = batch_norm(x, params, training)

    if stride == 1 and num_inp_filters == num_out_filters:
      with tf.variable_scope('residual'):
        x = stochastic_depth(x, training, stochastic_depth_drop_rate)
        x = x + residual
    log_tensor(x, True)
  return x


def stochastic_depth(x, training, drop_rate):
  """Implements the paper https://arxiv.org/pdf/1603.09382.pdf."""
  if not training:
    return x

  # Compute tensor.
  keep_rate = 1. - drop_rate
  batch_size = tf.shape(x)[0]
  random_tensor = tf.cast(keep_rate, x.dtype)
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=x.dtype)
  binary_tensor = tf.floor(random_tensor)
  x = tf.math.divide(x, keep_rate) * binary_tensor
  return x        
