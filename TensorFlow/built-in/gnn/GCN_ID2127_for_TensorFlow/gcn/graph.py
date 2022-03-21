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
# Author: Tao Wu (taowu1@huawei.com)

""" graph.py """

import tensorflow as tf

class GraphConvolutionLayer(tf.Module):
    """ basic GCN layer """
    def __init__(self, input_dim, output_dim, activ=None, use_bias=True,
                 sparse_input=False, sparse_adj=True):
        super(GraphConvolutionLayer, self).__init__()
        self.kernel = tf.Variable(glorot_uniform([input_dim, output_dim]),
                                  shape=[input_dim, output_dim], name='kernel')
        self.use_bias = use_bias
        self.bias = tf.Variable(tf.zeros([1, output_dim]),
                                shape=[1, output_dim], name='bias') if self.use_bias else None
        self.activ = activ
        self.sparse_input = sparse_input
        self.sparse_adj = sparse_adj

    def __call__(self, x):  # pylint: disable=invalid-name
        """ forward call """
        feat, adj = x
        if self.sparse_input:
            out = tf.sparse.sparse_dense_matmul(feat, self.kernel)
        else:
            out = tf.matmul(feat, self.kernel)
        if self.sparse_adj:
            out = tf.sparse.sparse_dense_matmul(adj, out)
        else:
            out = tf.matmul(adj, out)
        if self.use_bias:
            out = tf.add(out, self.bias)
        if self.activ is not None:
            out = self.activ(out)
        return out


class GraphConvolutionModel(tf.Module):
    """ GCN model """
    def __init__(self, input_dim, output_dim, hidden_dim, keep_prob=0.5,
                 dropout_fn=None, sparse_input=True, sparse_adj=True, model_name='GCN'):
        super(GraphConvolutionModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout_fn if dropout_fn is not None else tf.nn.dropout
        self.keep_prob = keep_prob
        self.sparse_input = sparse_input
        self.sparse_adj = sparse_adj
        self.model_name = model_name
        with tf.variable_scope('Weights/graph_conv_layer_1', reuse=False):
            self.graph_conv_1 = GraphConvolutionLayer(
                self.input_dim, self.hidden_dim, activ=tf.nn.relu,
                sparse_input=self.sparse_input, sparse_adj=self.sparse_adj)
        with tf.variable_scope('Weights/graph_conv_layer_2', reuse=False):
            self.graph_conv_2 = GraphConvolutionLayer(
                self.hidden_dim, self.output_dim,
                sparse_input=False, sparse_adj=self.sparse_adj)

    def __call__(self, x, training=False):  # pylint: disable=invalid-name
        """ forward call """
        with tf.name_scope(self.model_name):
            _, adj = x
            hidden = self.graph_conv_1(x)
            if training and self.dropout is not None:
                hidden = self.dropout(hidden, keep_prob=self.keep_prob)
            logits = self.graph_conv_2((hidden, adj))
        return logits


def glorot_uniform(shape):
    """ glorot uniform """
    bound = tf.math.sqrt(6. / (shape[0] + shape[1]))
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=tf.float32)
