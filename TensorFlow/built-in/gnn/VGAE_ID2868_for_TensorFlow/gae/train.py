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

from __future__ import division, print_function

from datetime import datetime
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from hyperparameters import FLAGS
from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

if tf.__version__[0] != '1':
    tf.disable_eager_execution()
if FLAGS.device == 'npu':
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
logdir = None
if FLAGS.profiling_gpu:
    from tensorflow import profiler
    logdir = "./tf_logs/run_{}/".format(datetime.now().strftime("%Y%m%d%H%M%S"))

def config_device(config=None):
    """configure device"""
    if FLAGS.device == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)
    elif FLAGS.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    elif FLAGS.device == 'npu':
        config = tf.ConfigProto() if config is None else config
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["profiling_mode"].b = FLAGS.profiling_npu
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(FLAGS.profiling_options)
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    return config

CONFIG = config_device()
dropout_fn = tf.nn.dropout if FLAGS.device != 'npu' else npu_ops.dropout
if not os.path.exists(FLAGS.out_path):
    os.mkdir(FLAGS.out_path)
logs_path = os.path.join(FLAGS.out_path, 'train_' + FLAGS.dataset + '_' + FLAGS.device + '_.log')
# logs = open(logs_path, 'w')
logs = None

# Load data
adj, features = load_data(FLAGS.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

num_nodes, num_features = features[2]
features_nonzero = features[1].shape[0]
adj_norm_nonzero = adj_norm[1].shape[0]
adj_label_nonzero = adj_label[1].shape[0]

# Save inputs
if FLAGS.save_inputs:
    npy_path = os.path.join(FLAGS.out_path, 'inputs_' + FLAGS.dataset + '.npy')
    if os.path.exists(npy_path):
        os.remove(npy_path)
    np.save(npy_path, {
        'features': features,
        'adj_orig': adj_orig,
        'adj_norm': adj_norm,
        'adj_label': adj_label,
        'val_edges': val_edges,
        'val_edges_false': val_edges_false,
        'test_edges': test_edges,
        'test_edges_false': test_edges_false,
    }, allow_pickle=True)

# Create default graph and session
G = tf.Graph()
sess = tf.Session(graph=G, config=CONFIG)
with G.as_default():

    # Define placeholders
    with tf.variable_scope('Placeholders'):
        placeholders = {
            'features_val': tf.placeholder(tf.float32, (features_nonzero, ), name='features/values'),
            'features_idx': tf.placeholder(tf.int64, (features_nonzero, 2), name='features/indices'),
            'features_shape': tf.constant(features[2], tf.int64, name='features/shape'),
            'adj_val': tf.placeholder(tf.float32, (adj_norm_nonzero, ), name='adj/values'),
            'adj_idx': tf.placeholder(tf.int64, (adj_norm_nonzero, 2), name='adj/indices'),
            'adj_shape': tf.constant(adj_norm[2], tf.int64, name='adj/shape'),
            'adj_orig_val': tf.placeholder(tf.float32, (adj_label_nonzero, ), name='adj_orig/values'),
            'adj_orig_idx': tf.placeholder(tf.int64, (adj_label_nonzero, 2), name='adj_orig/indcies'),
            'adj_orig_shape': tf.constant(adj_label[2], tf.int64, name='adj_orig/shape'),
            'dropout': tf.placeholder(tf.float32, (), name='dropout'),
            'edges_pos': tf.placeholder(tf.int32, (None, 2), name='edges_pos'),
            'edges_neg': tf.placeholder(tf.int32, (None, 2), name='edges_neg'),
        }
        placeholders['features'] = tf.sparse.SparseTensor(
            placeholders['features_idx'], placeholders['features_val'], placeholders['features_shape'])
        placeholders['adj'] = tf.sparse.SparseTensor(
            placeholders['adj_idx'], placeholders['adj_val'], placeholders['adj_shape'])
        placeholders['adj_orig'] = tf.sparse.SparseTensor(
            placeholders['adj_orig_idx'], placeholders['adj_orig_val'], placeholders['adj_orig_shape'])

    # Create model
    model = None
    if FLAGS.model == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero, dropout_fn)
    elif FLAGS.model == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, dropout_fn)

    # Optimizer
    with tf.name_scope('Optimizer'):
        use_dropout = (FLAGS.dropout > 0.)
        if FLAGS.model == 'gcn_ae':
            opt = OptimizerAE(model=model,
                              labels=tf.reshape(tf.sparse.to_dense(
                                  placeholders['adj_orig'], validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm,
                              use_dropout=use_dropout)
        elif FLAGS.model == 'gcn_vae':
            opt = OptimizerVAE(model=model,
                               labels=tf.reshape(tf.sparse.to_dense(
                                   placeholders['adj_orig'], validate_indices=False), [-1]),
                               num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               use_dropout=use_dropout)

    # Validation
    with tf.name_scope('Validation'):
        adj_recon = model.reconstruct()
        preds_pos = tf.math.sigmoid(tf.gather_nd(adj_recon, placeholders['edges_pos']), name='preds_pos')
        preds_neg = tf.math.sigmoid(tf.gather_nd(adj_recon, placeholders['edges_neg']), name='preds_neg')
        preds_all = tf.concat([preds_pos, preds_neg], axis=0, name='preds_all')
        labels_all = tf.concat([tf.ones_like(preds_pos), tf.zeros_like(preds_neg)], axis=0, name='labels_all')

    # Initialization and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables())

def get_roc_score(edges_pos, edges_neg):
    """ get ROC & AP scores """
    feed_dict.update({
        placeholders['dropout']: 0,
        placeholders['edges_pos']: edges_pos,
        placeholders['edges_neg']: edges_neg,
    })
    preds_all_, labels_all_= sess.run([preds_all, labels_all], feed_dict=feed_dict)
    roc_score = roc_auc_score(labels_all_, preds_all_)
    ap_score = average_precision_score(labels_all_, preds_all_)
    return roc_score, ap_score

# Train model
feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
val_roc_score = []
val_ap_score = []
timer = datetime.now()
if FLAGS.profiling_gpu:
    _ = tf.summary.FileWriter(logdir, graph=G)
    profiler.experimental.start(logdir)
for epoch in range(FLAGS.epochs):
    # Execute training step
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    _, avg_cost, avg_accuracy = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Validate training
    roc_new, ap_new = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_new)
    val_ap_score.append(ap_new)

    # Print metrics
#     timer_ = str(datetime.now() - timer)[:-4]
    timer_ = (datetime.now() - timer).total_seconds()
    print("ep= {} |train_loss= {:.6g} |train_acc= {:.6f} |val_roc= {:.6f} "
          "|val_ap= {:.6f} |epochs_per_second= {:.6g}".format(
              epoch + 1, avg_cost, avg_accuracy, val_roc_score[-1],
              val_ap_score[-1], timer_ / (epoch + 1)), file=logs)
print("Training ended. Total elapsed time in seconds: {}".format(timer_), file=logs)
if FLAGS.profiling_gpu:
    profiler.experimental.stop()


def final_test():
    """Print final test ROC & AP scores"""
    feed_dict.update({
        placeholders['dropout']: 0,
        placeholders['edges_pos']: test_edges,
        placeholders['edges_neg']: test_edges_false,
    })
    preds_pos_, preds_neg_ = sess.run([preds_pos, preds_neg], feed_dict=feed_dict)
    preds_all_ = np.concatenate([preds_pos_, preds_neg_], axis=0)
    labels_all_ = np.concatenate([np.ones_like(preds_pos_), np.zeros_like(preds_neg_)], axis=0)
    roc_ = roc_auc_score(labels_all_, preds_all_)
    ap_ = average_precision_score(labels_all_, preds_all_)
    return roc_, ap_

roc_final, ap_final = final_test()
print('Test ROC score: {:.6f}'.format(roc_final), file=logs)
print('Test AP score: {:.6f}'.format(ap_final), file=logs)

def plot_metrics(metrics, out_path='./saved/metrics.png', ylabel='score', figsize=(12, 12)):
    """plot metrics"""
    fig, axes = plt.subplots(figsize=figsize)
    axes.plot(range(1, len(metrics) + 1), metrics)
    axes.set(xlabel='epochs', ylabel=ylabel)
    fig.savefig(out_path, format='png')
    plt.close(fig)
    print("Save metrics plot to: {}".format(out_path))

plot_metrics(val_roc_score, out_path='./saved/val_roc.png', ylabel='val roc score')
plot_metrics(val_ap_score, out_path='./saved/vald_ap.png', ylabel='val ap score')

def save_to_pb():
    """save constant graph to pb"""
    ckpt_path = os.path.join(FLAGS.out_path, FLAGS.ckpt_file + FLAGS.dataset)
    saver.save(sess, ckpt_path, global_step=None)
    print("Save checkpoint to: {}".format(ckpt_path))
    pb_path = os.path.join(FLAGS.out_path, FLAGS.pb_file + FLAGS.dataset + '.pb')
    constant_graph = convert_variables_to_constants(
        sess, sess.graph_def, ['Validation/preds_all', 'Validation/labels_all'])
    with tf.gfile.FastGFile(pb_path, mode='wb') as file:
        file.write(constant_graph.SerializeToString())
    print('Write constant graph to: {}'.format(pb_path))

save_to_pb()
sess.close()
if logs is not None:
    logs.close()
