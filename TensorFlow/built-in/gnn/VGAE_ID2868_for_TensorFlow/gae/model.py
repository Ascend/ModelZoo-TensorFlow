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
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, sparse_dropout
from hyperparameters import FLAGS

# pylint:disable=missing-function-docstring
class GCNModelAE(tf.Module):
    """GCN-AE model"""
    def __init__(self, placeholders, num_features, features_nonzero, dropout_fn=None):
        super(GCNModelAE, self).__init__()
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.dropout_fn = tf.nn.dropout if dropout_fn is None else dropout_fn
        with tf.variable_scope('GAE_weights'):
            self.GraphConvolutionSparse = GraphConvolutionSparse(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1, act=tf.nn.relu)
            self.GraphConvolution = GraphConvolution(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2, act=lambda x: x)
            self.InnerProductDecoder = InnerProductDecoder(act=lambda x: x)

    def __call__(self, inputs, use_dropout=True):
        if use_dropout:
            inputs = sparse_dropout(inputs, (self.features_nonzero, ), 1 - self.dropout)
        hidden1 = self.GraphConvolutionSparse((inputs, self.adj))
        if use_dropout:
            hidden1 = self.dropout_fn(hidden1, keep_prob=1 - self.dropout)
        embedding = self.GraphConvolution((hidden1, self.adj))
        adj_recon = self.InnerProductDecoder(embedding)
        return adj_recon

    def reconstruct(self):
        """ reconstruct for inference """
        hidden1 = self.GraphConvolutionSparse((self.inputs, self.adj))
        embedding = self.GraphConvolution((hidden1, self.adj))
        adj_recon = tf.matmul(embedding, embedding, transpose_b=True)
        return adj_recon


class GCNModelVAE(tf.Module):
    """GCN-VAE model"""
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, dropout_fn=None):
        super(GCNModelVAE, self).__init__()
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = num_nodes ** 2
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.dropout_fn = tf.nn.dropout if dropout_fn is None else dropout_fn
        with tf.variable_scope('VGAE_weights'):
            self.GraphConvolutionSparse = GraphConvolutionSparse(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1, act=tf.nn.relu)
            self.GraphConvolution_1 = GraphConvolution(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2, act=lambda x: x)
            self.GraphConvolution_2 = GraphConvolution(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2, act=lambda x: x)
            self.InnerProductDecoder = InnerProductDecoder(
                output_dim=self.output_dim, act=lambda x: x)

    def __call__(self, inputs, use_dropout=True):
        if use_dropout:
            inputs = sparse_dropout(inputs, (self.features_nonzero, ), 1 - self.dropout)
        hidden = self.GraphConvolutionSparse((inputs, self.adj))
        if use_dropout:
            hidden1 = self.dropout_fn(hidden, keep_prob=1 - self.dropout)
            hidden2 = self.dropout_fn(hidden, keep_prob=1 - self.dropout)
            z_mean = self.GraphConvolution_1((hidden1, self.adj))
            z_log_std = self.GraphConvolution_2((hidden2, self.adj))
        else:
            z_mean = self.GraphConvolution_1((hidden, self.adj))
            z_log_std = self.GraphConvolution_2((hidden, self.adj))
        z = z_mean + tf.random.normal([self.n_samples, FLAGS.hidden2]) * tf.exp(z_log_std)
        adj_recon = self.InnerProductDecoder(z)
        return adj_recon, z_mean, z_log_std

    def reconstruct(self):
        """ reconstruct for inference """
        assert isinstance(self.inputs, tf.sparse.SparseTensor), "Input has to be a tf.sparse.SparseTensor!"
        hidden = self.GraphConvolutionSparse((self.inputs, self.adj))
        embedding = self.GraphConvolution_1((hidden, self.adj))
        adj_recon = tf.matmul(embedding, embedding, transpose_b=True)
        return adj_recon
