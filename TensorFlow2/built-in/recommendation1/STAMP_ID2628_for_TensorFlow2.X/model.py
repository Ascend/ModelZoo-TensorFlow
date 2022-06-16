#
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
#
"""
Created on Oct 23, 2020

model: STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation

@author: Ziyao Geng
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, \
    Dropout, Embedding, Flatten, Input

from modules import *


class STAMP(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, item_pooling, maxlen=40, activation='tanh', embed_reg=1e-4):
        """
        STAMP
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param item_pooling: A Ndarray or Tensor, shape=(m, n),
        m is the number of items, and n is the number of behavior feature. The item pooling.
        :param activation: A String. The activation of FFN.
        :param maxlen: A scalar. Maximum sequence length.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(STAMP, self).__init__()
        # maximum sequence length
        self.maxlen = maxlen

        # item pooling
        self.item_pooling = item_pooling
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        # if behavior feature list contains itemId and item category id, seq_len = 2
        self.seq_len = len(behavior_feature_list)

        # embedding dim, each sparse feature embedding dimension is the same
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # Attention
        self.attention_layer = Attention_Layer(d=self.embed_dim)

        # FNN, hidden unit must be equal to embedding dimension
        self.ffn1 = Dense(self.embed_dim, activation=activation)
        self.ffn2 = Dense(self.embed_dim, activation=activation)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        dense_inputs, sparse_inputs, seq_inputs = inputs
        
        x = dense_inputs
        # other
        for i in range(self.other_sparse_len):
            x = tf.concat([x, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq
        seq_embed, m_t, item_pooling_embed = None, None, None
        for i in range(self.seq_len):
            # item sequence embedding
            seq_embed = self.embed_seq_layers[i](seq_inputs[:, i]) if seq_embed is None \
                else seq_embed + self.embed_seq_layers[i](seq_inputs[:, i])
            # last click item embedding
            m_t = self.embed_seq_layers[i](seq_inputs[:, i, -1]) if m_t is None \
                else m_t + self.embed_seq_layers[i](seq_inputs[-1, i, -1])  # (None, d)
            # item pooling embedding 
            item_pooling_embed = self.embed_seq_layers[i](self.item_pooling[:, i]) \
                if item_pooling_embed is None \
                else item_pooling_embed + self.embed_seq_layers[i](self.item_pooling[:, i])  # (m, d)

        # calculate m_s        
        m_s = tf.reduce_mean(seq_embed, axis=1)  # (None, d)

        # attention
        m_a = self.attention_layer([seq_embed, m_s, m_t])  # (None, d)
        # if model is STMP, m_a = m_s
        # m_a = m_s

        # try to add other embedding vector
        if self.other_sparse_len != 0 or self.dense_len != 0:
            m_a = tf.concat([m_a, x], axis=-1)
            m_t = tf.concat([m_t, x], axis=-1)

        # FFN
        h_s = self.ffn1(m_a)  # (None, d)
        h_t = self.ffn2(m_t)  # (None, d)

        # Calculate
        # h_t * item_pooling_embed, (None, 1, d) * (m, d) = (None, m, d)
        # () mat h_s, (None, m, d) matmul (None, d, 1) = (None, m, 1)
        z = tf.matmul(tf.multiply(tf.expand_dims(h_t, axis=1), item_pooling_embed), tf.expand_dims(h_s, axis=-1))
        z = tf.squeeze(z, axis=-1)  # (None, m)

        # Outputs
        outputs = tf.nn.softmax(z)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.seq_len, self.maxlen), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs])).summary()


def test_model():
    dense_features = []  # [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    item_pooling = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    features = [dense_features, sparse_features]
    model = STAMP(features, behavior_list, item_pooling)
    model.summary()


# test_model()