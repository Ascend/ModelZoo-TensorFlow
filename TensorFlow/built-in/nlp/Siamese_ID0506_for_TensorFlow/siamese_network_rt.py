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
import tensorflow as tf
import numpy as np
from npu_bridge.estimator.npu.npu_dynamic_rnn import DynamicRNN


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden = hidden_units
        n_layers = 3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def BiRNN_npu(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden = hidden_units
        # n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.transpose(x, perm=[1, 0, 2], name="transpose_inputdata")
        print(x)

        # dropout_rate is 1, so not add dropout
        with tf.name_scope(scope), tf.variable_scope(scope):
            fw_cell1 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            fw_y1, output_h, output_c, i, j, f, o, tanhc = fw_cell1(x)
            bw_cell1 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            bw_y1, output_h, output_c, i, j, f, o, tanhc = bw_cell1(tf.reverse(x, axis=[0]))
            output_rnn1 = tf.concat((fw_y1, tf.reverse(bw_y1, axis=[0])), axis=2)

            fw_cell2 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            fw_y2, output_h, output_c, i, j, f, o, tanhc = fw_cell2(output_rnn1)
            bw_cell2 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            bw_y2, output_h, output_c, i, j, f, o, tanhc = bw_cell2(tf.reverse(output_rnn1, axis=[0]))
            output_rnn2 = tf.concat((fw_y2, tf.reverse(bw_y2, axis=[0])), axis=2)

            fw_cell3 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            fw_y3, output_h, output_c, i, j, f, o, tanhc = fw_cell3(output_rnn2)
            bw_cell3 = DynamicRNN(hidden_size=n_hidden, forget_bias=1.0, dtype=tf.float32)
            bw_y3, output_h, output_c, i, j, f, o, tanhc = bw_cell3(tf.reverse(output_rnn2, axis=[0]))
            output_rnn3 = tf.concat((fw_y3, tf.reverse(bw_y3, axis=[0])), axis=2)

            outputs = tf.transpose(output_rnn3, perm=[1, 0, 2], name="transpose_outdata")
            print(outputs)

        return outputs[:, -1, :]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(
            self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True, name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            # self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            # self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.BiRNN_npu(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size,
                                       sequence_length, hidden_units)
            self.out2 = self.BiRNN_npu(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size,
                                       sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            #self.distance = util.set_graph_exec_config(self.distance, dynamic_input=True,
                                                       #dynamic_graph_execute_mode='dynamic_execute',
                                                       #dynamic_inputs_shape_range='data:[64,15],[64,15],[64]')
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
