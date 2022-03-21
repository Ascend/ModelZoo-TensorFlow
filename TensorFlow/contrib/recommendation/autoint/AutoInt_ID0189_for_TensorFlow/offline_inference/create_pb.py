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
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops
from model import multihead_attention

ckpt_path = 'pre_trained/ml-1m/model.ckpt'
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main():
    tf.reset_default_graph()

    field_size = 7
    output_size = 64
    blocks = 3

    feat_index = tf.placeholder(tf.int32, shape=(None, field_size-1), name='feat_index')
    feat_value = tf.placeholder(tf.float32, shape=(None, field_size-1), name='feat_value')
    genre_index = tf.placeholder(tf.int32, shape=(None, field_size-1), name='genre_index')
    genre_value = tf.placeholder(tf.float32, shape=(None, field_size-1), name='genre_value')

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

    y_deep = embeddings
    for i in range(blocks):
        y_deep = multihead_attention(queries=y_deep,
                                     keys=y_deep,
                                     values=y_deep,
                                     num_units=64,
                                     num_heads=2,
                                     dropout_keep_prob=1.0,
                                     is_training=False,
                                     has_residual=True)

    flat = tf.reshape(y_deep,
                      shape=[-1, output_size * field_size])

    out = tf.add(tf.matmul(flat, weights["prediction"]),
                 weights["prediction_bias"], name='logits')
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
            output_graph='save/autoint.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

def _initialize_weights():
    weights = dict()
    feature_size = 3600
    field_size = 7
    embedding_size = 16
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
