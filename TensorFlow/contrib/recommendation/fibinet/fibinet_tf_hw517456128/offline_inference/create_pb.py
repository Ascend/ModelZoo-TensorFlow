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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph
from model import sparse_linear, SENET_layer, Bilinear_layer, stack_dense_layer
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops

ckpt_path = 'pre_trained/ml-1m/model.ckpt'
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main():
    tf.reset_default_graph()

    feature_size = 3600
    field_size = 7
    embedding_size = 32

    feat_index = tf.placeholder(tf.int32, shape=(None, 6), name='feat_index')
    feat_value = tf.placeholder(tf.float32, shape=(None, 6), name='feat_value')
    genre_index = tf.placeholder(tf.int32, shape=(None, 6), name='genre_index')
    genre_value = tf.placeholder(tf.float32, shape=(None, 6), name='genre_value')

    weights = _initialize_weights()

    embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)
    feat_value = tf.reshape(feat_value, shape=[-1, field_size-1, 1])
    embeddings = tf.multiply(embeddings, feat_value)

    embeddings_m = tf.nn.embedding_lookup(weights['feature_embeddings'], genre_index)
    genre_value = tf.reshape(genre_value, shape=[-1, 6, 1])
    embeddings_m = tf.multiply(embeddings_m, genre_value)
    embeddings_m = tf.reduce_sum(embeddings_m, axis=1)
    embeddings_m = tf.div(embeddings_m, tf.reduce_sum(genre_value, axis=1, keep_dims=False))

    embeddings = tf.concat([embeddings, tf.expand_dims(embeddings_m, 1)], 1)
    embeddings = tf.nn.dropout(embeddings, 1.0)

    linear_output = sparse_linear(feature_size, feat_index, feat_value, genre_index, genre_value, add_summary=False)
    senet_embedding_matrix = SENET_layer(embeddings, field_size, embedding_size, pool_op='mean', ratio=3)
    bi_org = Bilinear_layer(embeddings, field_size, embedding_size, type='field_all', name='org')
    bi_senet = Bilinear_layer(senet_embedding_matrix, field_size, embedding_size, type='field_all', name='senet')

    combination_layer = tf.concat([bi_org, bi_senet], axis=1)

    dense_output = stack_dense_layer(combination_layer, [128, 128], 0.1, True, mode=False, add_summary=False)
    dense_output = tf.layers.dense(dense_output, units=1, activation=None, name='dense_final')

    print(dense_output.shape, linear_output.shape)
    with tf.variable_scope('output'):
        out = dense_output + linear_output

    out = tf.nn.sigmoid(out, name='pred')

    with tf.Session(config=config) as sess:
        tf.train.write_graph(sess.graph_def, 'save', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='save/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='pred',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='save/fibinet.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

def _initialize_weights():
    weights = dict()
    feature_size = 3600
    field_size = 7
    embedding_size = 32
    output_size = 64

    weights["feature_embeddings"] = tf.get_variable(shape=[feature_size, embedding_size],
        initializer=tf.glorot_uniform_initializer(), name="feature_embeddings")  # feature_size(n) * d

    input_size = output_size * field_size

    glorot = np.sqrt(2.0 / (input_size + 1))
    weights["prediction"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32, name="prediction")
    weights["prediction_bias"] = tf.Variable(
                        np.random.normal(), dtype=np.float32, name="prediction_bias")

    return weights

if __name__ == '__main__':
    main()
