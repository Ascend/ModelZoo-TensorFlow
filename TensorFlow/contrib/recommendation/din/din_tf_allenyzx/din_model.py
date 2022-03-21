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
from modules import Attention_Layer, Dense, Dice, PReLU, Embedding
from tensorflow.nn import l2_normalize
from tensorflow.keras.regularizers import L1L2
from tensorflow.python.ops.init_ops import RandomNormal as TFRandomNormal
from tensorflow.python.framework import dtypes
from tensorflow.layers import BatchNormalization


class RandomNormal(TFRandomNormal):
    """Initializer that generates tensors with a normal distribution.

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate. Defaults to 0.
      stddev: a python scalar or a scalar tensor. Standard deviation of the random
        values to generate. Defaults to 0.05.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: The data type. Only floating point types are supported.

    Returns:
        RandomNormal instance.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=dtypes.float32):
        super(RandomNormal, self).__init__(
            mean=mean, stddev=stddev, seed=seed, dtype=dtype)


class DIN:
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='prelu', ffn_activation='prelu', maxlen=40, dnn_dropout=0.,
                 embed_reg=1e-4, device='npu'):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param device: String. choose cpu, gpu or npu.
        """
        self.maxlen = maxlen
        self.dnn_dropout = dnn_dropout

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer=RandomNormal(),
                                              embeddings_regularizer=L1L2(l2=embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer=RandomNormal(),
                                           embeddings_regularizer=L1L2(l2=embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)
        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]

        self.device = device

        if self.dnn_dropout != 0:
            if device != 'npu':
                from tensorflow.layers import Dropout
                self.dropout = Dropout(dnn_dropout)
            else:
                from npu_bridge.estimator import npu_ops
                self.dropout = npu_ops.dropout

        self.dense_final = Dense(1)

    def __call__(self, inputs):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)
        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)],
                              axis=-1)
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)

        # att
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, d * 2)

        # concat user_info(att hist), cadidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        if self.dnn_dropout != 0:
            if self.device != 'npu':
                info_all = self.dropout(info_all)
            else:
                info_all = self.dropout(info_all, keep_prob=self.dnn_dropout)

        outputs = tf.nn.sigmoid(self.dense_final(info_all),name='output')
        return outputs
