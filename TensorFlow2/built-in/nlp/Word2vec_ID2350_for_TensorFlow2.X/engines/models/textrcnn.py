# -*- coding: utf-8 -*-
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
# @Time : 2020/11/6 10:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textrcnn.py
# @Software: PyCharm
# import npu_device
# npu_device.open().as_default()
from abc import ABC
import tensorflow as tf
from config import classifier_config
import npu_convert_dropout

class TextRCNN(tf.keras.Model, ABC):
    """
    TextRCNN模型
    """

    def __init__(self, seq_length, num_classes, hidden_dim, embedding_dim, vocab_size):
        super(TextRCNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding_method = classifier_config['embedding_method']
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, self.embedding_dim, mask_zero=True)

        self.forward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.backward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, go_backwards=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense1 = tf.keras.layers.Dense(2 * self.hidden_dim + self.embedding_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')

    @tf.function
    def call(self, inputs, training=None):
        # 不引入外部的embedding
        if self.embedding_method is None:
            inputs = self.embedding(inputs)

        left_embedding = self.forward(inputs)
        right_embedding = self.backward(inputs)
        concat_outputs = tf.keras.layers.concatenate([left_embedding, inputs, right_embedding], axis=-1)
        dropout_outputs = self.dropout(concat_outputs, training)
        fc_outputs = self.dense1(dropout_outputs)
        pool_outputs = self.max_pool(fc_outputs)
        outputs = self.dense2(pool_outputs)
        return outputs

