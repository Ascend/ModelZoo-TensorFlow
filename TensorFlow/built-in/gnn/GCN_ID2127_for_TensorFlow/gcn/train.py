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

""" train.py """

from __future__ import print_function

import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.python.framework.graph_util \
    import convert_variables_to_constants  # pylint: disable=import-error

from graph import GraphConvolutionModel
from dataset import CoraData
from hyperparameters import FLAGS
# pylint: disable=import-error, unused-import, ungrouped-imports
if FLAGS.device == 'npu':
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# pylint: disable=invalid-name
def train(data, config=None):
    """ train GCN """
    hidden_dim = FLAGS.hidden_dim
    l2_regularizer = FLAGS.l2_regularizer
    keep_prob = FLAGS.keep_prob
    num_epochs = FLAGS.num_epochs
    display_per_epochs = FLAGS.display_per_epochs
    learning_rate = FLAGS.learning_rate
    patience = FLAGS.patience
    sparse_input = FLAGS.sparse_input
    sparse_adj = FLAGS.sparse_adj
    out_path = FLAGS.out_path
    ckpt_file = FLAGS.ckpt_file
    pb_file = FLAGS.pb_file
    device = FLAGS.device
    dropout_fn = tf.nn.dropout if device != 'npu' else npu_ops.dropout
    if not os.path.exists(out_path):
        os.mkdir(out_path)
#     logs = open(os.path.join(out_path, 'train.log'), 'w')
    logs = None
    print("device= {} |sparse_input= {} |sparse_adj= {} |hidden_dim= {} |"
          "keep_prob= {:.2f} |l2_regu= {:.1e} |lr= {:.1e}".format(
              device, sparse_input, sparse_adj, hidden_dim,
              keep_prob, l2_regularizer, learning_rate), file=logs)

    features, labels, adjacency = data.features, data.labels, data.adjacency
    input_dim, output_dim = data.feature_dim, data.num_classes
    X_data, X_idx, X_nnz, A_data, A_idx, A_nnz = convert_data(
        features, adjacency, sparse_input, sparse_adj)

    G = tf.Graph()
    sess = tf.Session(graph=G, config=config)
    with G.as_default():
        with tf.variable_scope('Placeholders', reuse=True):
            placeholders = {}
            placeholders['y'] = tf.placeholder(tf.int32, labels.shape, name='labels')
            placeholders['mask_train'] = tf.placeholder(tf.int32, (None,), name='mask_train')
            placeholders['mask_valid'] = tf.placeholder(tf.int32, (None,), name='mask_valid')
            if sparse_input:
                placeholders['X_data'] = tf.placeholder(tf.float32, (X_nnz,), name='X_data')
                placeholders['X_idx'] = tf.placeholder(tf.int64, (X_nnz, 2), name='X_idx')
                placeholders['X_shape'] = features.shape
            else:
                placeholders['X_data'] = tf.placeholder(tf.float32, features.shape, name='X_data')
                placeholders['X_idx'] = tf.placeholder(tf.int64, (), name='X_idx')
            if sparse_adj:
                placeholders['A_data'] = tf.placeholder(tf.float32, (A_nnz,), name='A_data')
                placeholders['A_idx'] = tf.placeholder(tf.int64, (A_nnz, 2), name='A_idx')
                placeholders['A_shape'] = adjacency.shape
            else:
                placeholders['A_data'] = tf.placeholder(tf.float32, adjacency.shape, name='A_data')
                placeholders['A_idx'] = tf.placeholder(tf.int64, (), name='A_idx')

        X_sp = sparse_dropout(
            (placeholders['X_idx'], placeholders['X_data'], placeholders['X_shape']),
            (X_nnz, ), keep_prob) if sparse_input else \
            dropout_fn(placeholders['X_data'], keep_prob=keep_prob)
        A_sp = tf.sparse.SparseTensor(
            placeholders['A_idx'], placeholders['A_data'], placeholders['A_shape']) \
            if sparse_adj else placeholders['A_data']
        model = GraphConvolutionModel(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
            keep_prob=keep_prob, dropout_fn=dropout_fn,
            sparse_input=sparse_input, sparse_adj=sparse_adj)
        logits = model((X_sp, A_sp), training=True)
        masked_logits = tf.gather(logits, placeholders['mask_train'], name='masked_logits')
        masked_labels = tf.gather(
            placeholders['y'], placeholders['mask_train'], name='masked_labels')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=masked_labels, logits=masked_logits)
        loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        if l2_regularizer != 0:
            l2_loss = tf.add(tf.nn.l2_loss(model.graph_conv_1.kernel),
                             tf.nn.l2_loss(model.graph_conv_2.kernel), name='l2_loss')
            loss_op = tf.add(loss_op, l2_loss * l2_regularizer, name='combined_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, var_list=tf.trainable_variables())

        X_sp = tf.sparse.SparseTensor(
            placeholders['X_idx'], placeholders['X_data'], placeholders['X_shape']) \
            if sparse_input else placeholders['X_data']
        masked_logits2 = tf.gather(model((X_sp, A_sp), training=False),
                                   placeholders['mask_valid'], name='masked_logits_2')
        masked_labels2 = tf.gather(
            placeholders['y'], placeholders['mask_valid'], name='masked_labels_2')
        cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=masked_labels2, logits=masked_logits2)
        loss_op2 = tf.reduce_mean(cross_entropy2, name='cross_entropy_loss_2')
        accuracy2 = tf.reduce_mean(tf.cast(tf.math.equal(
            masked_labels2, tf.math.argmax(masked_logits2, -1, output_type=tf.dtypes.int32)),
                                          dtype=tf.float32), name='accuracy_2')

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

    # Training loop
    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'valid_acc': [],
        'test_acc': [],
    }
    epoch, inpatience, best_valid_loss = 0, 0, 99999.
    ###########################################################################
    start_time = datetime.now()
    for epoch in range(1, num_epochs + 1):
        timer = datetime.now()
        loss_train, _ = sess.run([loss_op, train_op],
                                 feed_dict={placeholders['X_data']: X_data,
                                            placeholders['X_idx']: X_idx,
                                            placeholders['A_data']: A_data,
                                            placeholders['A_idx']: A_idx,
                                            placeholders['y']: labels,
                                            placeholders['mask_train']: data.train_mask})
        timer_ = (datetime.now() - timer).total_seconds()

        loss_valid, acc_valid = sess.run([loss_op2, accuracy2],
                                         feed_dict={placeholders['X_data']: X_data,
                                                    placeholders['X_idx']: X_idx,
                                                    placeholders['A_data']: A_data,
                                                    placeholders['A_idx']: A_idx,
                                                    placeholders['y']: labels,
                                                    placeholders['mask_valid']: data.valid_mask})

        acc_test = sess.run(accuracy2,
                            feed_dict={placeholders['X_data']: X_data,
                                       placeholders['X_idx']: X_idx,
                                       placeholders['A_data']: A_data,
                                       placeholders['A_idx']: A_idx,
                                       placeholders['y']: labels,
                                       placeholders['mask_valid']: data.test_mask})
        
        metrics['train_loss'].append(loss_train)
        metrics['valid_loss'].append(loss_valid)
        metrics['valid_acc'].append(acc_valid)
        metrics['test_acc'].append(acc_test)
        if epoch % display_per_epochs == 0:
            print("epoch = {} | train_loss = {:.6g} | test_acc = {:.4f} | sec/step = {:.4g}".format(epoch, 
                                metrics['train_loss'][-1], metrics['test_acc'][-1], timer_), file=logs)
        if loss_valid < best_valid_loss:
            inpatience, best_valid_loss = 0, loss_valid
        else:
            if inpatience >= patience:
                print("ep {}: Early stopping activated.".format(epoch), file=logs)
                break
            inpatience += 1

    print("Training ended after {} epochs. Time elapsed: {}".format(
        epoch, (datetime.now() - start_time).total_seconds()), file=logs)
    ###########################################################################

    ckpt_path = os.path.join(out_path, ckpt_file)
    saver.save(sess, ckpt_path, global_step=None)
    print("Save checkpoint to: {}".format(ckpt_path))
    pb_path = os.path.join(out_path, pb_file)
    constant_graph = convert_variables_to_constants(sess, sess.graph_def, ['masked_logits_2'])
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    print('Write constant graph to: {}'.format(pb_path))
    sess.close()
    if logs is not None:
        logs.close()
    return metrics


def convert_data(features, adjacency, sparse_input=True, sparse_adj=True):
    """ convert data to sparse format """
    if sparse_input:
        features = sp.coo_matrix(features)
        X_data = features.data
        X_idx = np.array([features.row, features.col]).T
        X_nnz = features.count_nonzero()
    else:
        X_data, X_idx, X_nnz = features, 0, None
    if sparse_adj:
        adjacency = sp.coo_matrix(adjacency)
        A_data = adjacency.data
        A_idx = np.array([adjacency.row, adjacency.col]).T
        A_nnz = adjacency.count_nonzero()
    else:
        A_data, A_idx, A_nnz = adjacency.toarray(), 0, None
    return [X_data, X_idx, X_nnz, A_data, A_idx, A_nnz]


def sparse_dropout(x_sp, noise_shape, keep_prob):
    """ sparse dropout """
    x_indices, x_values, x_shape = x_sp
    noise = tf.random.uniform(noise_shape) + keep_prob
    mask = tf.cast(tf.floor(noise), dtype=tf.bool)
    idx = tf.reshape(tf.where(mask), [-1, ])
    x_values = tf.gather(x_values, idx) * (1. / keep_prob)
    x_indices = tf.gather(x_indices, idx)
    return tf.sparse.SparseTensor(x_indices, x_values, x_shape)


def plot_metrics(metrics, out_path='.'):
    """ plot training metrics """
    epoch = len(metrics['train_loss'])
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('training metrics', fontsize=14)
    axes[0].set_xlabel("epoch", fontsize=14)
    axes[0].set_ylabel("loss", fontsize=14)
    axes[0].plot(range(0, epoch), metrics['train_loss'], label='train_loss')
    axes[0].plot(range(1, epoch + 1), metrics['valid_loss'], label='valid_loss')
    axes[0].legend()
    axes[1].set_xlabel("epoch", fontsize=14)
    axes[1].set_ylabel("accuracy", fontsize=14)
    axes[1].plot(range(1, epoch + 1), metrics['valid_acc'], label='valid_acc')
    axes[1].plot(range(1, epoch + 1), metrics['test_acc'], label='test_acc')
    axes[1].legend()
    fig_path = os.path.join(out_path, 'metrics.png')
    fig.savefig(fig_path, format='png')
    plt.close(fig)
    print("Plot training metrics to: {}".format(fig_path))


def config_device(config=None):
    """ configure device """
    if FLAGS.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)
    elif FLAGS.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    elif FLAGS.device == 'npu':
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["profiling_mode"].b = FLAGS.profiling_mode
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(FLAGS.profiling_options)
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return config


if __name__ == '__main__':
    DATA = CoraData(
        data_path=FLAGS.data_path, cora_full=FLAGS.cora_full,
        shuffle=FLAGS.shuffle, take_subgraphs=FLAGS.take_subgraphs,
        train_size=FLAGS.train_size, valid_size=FLAGS.valid_size, test_size=FLAGS.test_size,
        min_train_samples_per_class=FLAGS.min_train_samples,
        min_valid_samples_per_class=FLAGS.min_valid_samples,
        save_inputs=True, out_path=FLAGS.out_path)
    CONFIG = config_device()
    METRICS = train(DATA, CONFIG)
    plot_metrics(METRICS, FLAGS.out_path)
