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
# Author: Tao Wu 

import tensorflow.compat.v1 as tf
from initializations import weight_variable_glorot

# pylint:disable=missing-function-docstring
def sparse_dropout(x, noise_shape, keep_prob):
    """Dropout for sparse inputs."""
    with tf.name_scope('sparse_dropout'):
        if isinstance(x, tf.sparse.SparseTensor):
            x_indices, x_values, x_shape = x.indices, x.values, x.dense_shape
        else:
            x_indices, x_values, x_shape = x
        noise = tf.random.uniform(noise_shape) + keep_prob
        mask = tf.cast(tf.floor(noise), dtype=tf.bool)
        idx = tf.reshape(tf.where(mask), [-1])
        x_values = tf.gather(x_values, idx) * (1. / keep_prob)
        x_indices = tf.gather(x_indices, idx)
        outputs = tf.sparse.SparseTensor(x_indices, x_values, x_shape)
    return outputs


class GraphConvolution(tf.Module):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, act=None):
        super(GraphConvolution, self).__init__()
        with tf.variable_scope('GraphConvolution'):
            self.kernel = weight_variable_glorot(input_dim, output_dim)
        self.act = tf.nn.relu if act is None else act

    def __call__(self, inputs):
        x, adj = inputs
        x = tf.matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(tf.Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, act=None):
        super(GraphConvolutionSparse, self).__init__()
        with tf.variable_scope('GraphConvolutionSparse'):
            self.kernel = weight_variable_glorot(input_dim, output_dim)
        self.act = tf.nn.relu if act is None else act

    def __call__(self, inputs):
        x, adj = inputs
        x = tf.sparse.sparse_dense_matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(tf.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, act=None, output_dim=-1):
        super(InnerProductDecoder, self).__init__()
        self.act = act if act is not None else tf.nn.sigmoid
        self.output_dim = output_dim

    def __call__(self, inputs):
        x = tf.matmul(inputs, inputs, transpose_b=True)
        x = tf.reshape(x, [self.output_dim])
        outputs = self.act(x)
        return outputs
