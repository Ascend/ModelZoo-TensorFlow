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

from npu_bridge.npu_init import *
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
from tensorflow.python.training.moving_averages import assign_moving_average

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(
    factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


class GRUCellHiddenDropout(GRUCell):
    """GRU cell add dropout on every hidden state.
    """

    def __init__(self,
                 num_units,
                 hidden_keep_prob=1.0,
                 mode=tf.estimator.ModeKeys.TRAIN):
        self.hidden_keep_prob = hidden_keep_prob
        self.mode = mode
        super(GRUCellHiddenDropout, self).__init__(num_units)

    def call(self, inputs, state):
        new_h, new_h = super(GRUCellHiddenDropout, self).call(inputs, state)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            new_h = npu_ops.dropout(new_h, keep_prob=self.hidden_keep_prob)
        return new_h, new_h


class BasicLSTMCellHiddenDropout(BasicLSTMCell):
    """Basic LSTM cell add dropout on every hidden state.
    """

    def __init__(self,
                 num_units,
                 hidden_keep_prob=1.0,
                 mode=tf.estimator.ModeKeys.TRAIN):
        self.hidden_keep_prob = hidden_keep_prob
        self.mode = mode
        super(BasicLSTMCellHiddenDropout, self).__init__(num_units)

    def call(self, inputs, state):
        new_h, new_h = super(BasicLSTMCellHiddenDropout, self).call(inputs,
                                                                    state)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            new_h = npu_ops.dropout(new_h, keep_prob=self.hidden_keep_prob)
        return new_h, new_h


class LSTMCellHiddenDropout(LSTMCell):
    """LSTM cell add dropout on every hidden state.
    """

    def __init__(self,
                 num_units,
                 hidden_keep_prob=1.0,
                 mode=tf.estimator.ModeKeys.TRAIN):
        self.hidden_keep_prob = hidden_keep_prob
        self.mode = mode
        super(LSTMCellHiddenDropout, self).__init__(num_units)

    def call(self, inputs, state):
        new_h, new_h = super(LSTMCellHiddenDropout, self).call(inputs, state)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            new_h = npu_ops.dropout(new_h, keep_prob=self.hidden_keep_prob)
        return new_h, new_h


def convolution(inputs, filter_shape, use_bias=False, activation=None,
                name="convolution", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        bias_shape = [filter_shape[-1]]
        strides = [1, 1, 1, 1]
        kernel_ = tf.compat.v1.get_variable(
            "filter_", filter_shape, dtype=tf.float32, regularizer=regularizer,
            initializer=initializer_relu()
            if activation is not None else initializer())
        outputs = tf.nn.conv2d(inputs, kernel_, strides, "VALID")
        if use_bias:
            outputs += tf.compat.v1.get_variable("bias_",
                                       bias_shape,
                                       regularizer=regularizer,
                                       initializer=tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def recurrent(inputs, rnn_dimension, sequence_length=None, cell_type="gru",
              cell_hidden_keep_prob=1.0, mode=tf.estimator.ModeKeys.TRAIN,
              use_bidirectional=False, name="rnn", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        if cell_type == "lstm_basic":
            rnn_fw_cell = BasicLSTMCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
            rnn_bw_cell = BasicLSTMCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
        elif cell_type == "lstm":
            rnn_fw_cell = LSTMCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
            rnn_bw_cell = LSTMCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
        elif cell_type == "gru":
            rnn_fw_cell = GRUCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
            rnn_bw_cell = GRUCellHiddenDropout(
                rnn_dimension, cell_hidden_keep_prob, mode)
        else:
            raise TypeError("Wrong cell type: " + cell_type)
        if use_bidirectional:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                rnn_fw_cell, rnn_bw_cell, inputs, dtype=tf.float32,
                sequence_length=sequence_length)
        else:
            outputs, output_states = tf.nn.dynamic_rnn(
                rnn_fw_cell, inputs, dtype=tf.float32)
        return outputs, output_states


def highway(x, activation=None, scope="highway", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        T = tf.contrib.layers.fully_connected(x, int(x.shape[-1]),
                                              activation_fn=tf.sigmoid)
        H = tf.contrib.layers.fully_connected(x, int(x.shape[-1]),
                                              activation_fn=activation)
        return H * T + x * (1.0 - T)


def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    """Batch norm layer
    Args:
        x: Input tensor.
        train: Whether is training, should be a tf.constant with tf.bool dtype.
        eps: A small float number to avoid dividing by 0.
        decay: Decay to calculate the mean and variance over the entire train
            dataset between batches.
        affine: Whether linear transform is needed.
        name: A name for this operation.
    Returns:
        Tensor after batch norm
    """
    with tf.variable_scope(name, default_name='BatchNorm'):
        params_shape = x.shape.as_list()[-1]
        moving_mean = tf.compat.v1.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.compat.v1.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def _mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)),
                                           name='moments')
            with tf.control_dependencies(
                [assign_moving_average(moving_mean, mean, decay),
                 assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(train, _mean_var_with_update,
                                 lambda: (
                                     moving_mean, moving_variance))
        if affine:
            beta = tf.compat.v1.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.compat.v1.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
