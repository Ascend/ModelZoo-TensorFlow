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


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input

from .modules import CrossNetwork, DNN


class DCN(Model):
    def __init__(self, feature_columns, dcn_layer_num ,dnn_hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, cross_w_reg=1e-6, cross_b_reg=1e-6):
        """
        Deep&Cross Network
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cross_w_reg: A scalar. The regularizer of cross network.
        :param cross_b_reg: A scalar. The regularizer of cross network.
        """
        super(DCN, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.layer_num = dcn_layer_num
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(dnn_hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = sparse_embed
        # Cross Network
        cross_x = self.cross_network(x)
        # DNN
        dnn_x = self.dnn_network(x)
        # Concatenate
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()