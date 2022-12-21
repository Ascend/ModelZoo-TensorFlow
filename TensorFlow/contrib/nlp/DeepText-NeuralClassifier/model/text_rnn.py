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
# ==============================================================================

from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.contrib import rnn

from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper


class TextRNNEstimator(NPUEstimator):
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]

            index = data_processor.dict_names.index(feature_name)
            embedding = embedding_layer.get_vocab_embedding(
                feature_name, features["fixed_len_" + feature_name],
                len(data_processor.dict_list[index]), params["epoch"],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index],
                mode=mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                embedding = model_helper.dropout(
                    embedding,
                    config.embedding_layer.embedding_dropout_keep_prob)
            rnn_fw_cell, rnn_bw_cell = None, None
            if config.TextRNN.cell_type == "lstm":
                rnn_fw_cell = rnn.BasicLSTMCell(config.TextRNN.rnn_dimension)
                rnn_bw_cell = rnn.BasicLSTMCell(config.TextRNN.rnn_dimension)
            elif config.TextRNN.cell_type == "gru":
                rnn_fw_cell = rnn.GRUCell(config.TextRNN.rnn_dimension)
                rnn_bw_cell = rnn.GRUCell(config.TextRNN.rnn_dimension)
            if config.TextRNN.use_bidirectional:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_fw_cell, rnn_bw_cell, embedding, dtype=tf.float32,
                    sequence_length=tf.reshape(
                        features[feature_name + "_fixed_real_len"], [-1]))
                text_embedding = tf.concat(outputs, 2)
            else:
                text_embedding, _ = tf.nn.dynamic_rnn(
                    rnn_fw_cell, embedding, dtype=tf.float32)

            if config.model_common.use_self_attention:
                hidden_layer = model_helper.self_attention(
                    text_embedding, config.model_common.attention_dimension)
            else:
                sum_layer = tf.reduce_sum(text_embedding, axis=1)
                hidden_layer = sum_layer / tf.cast(
                    features[feature_name + "_fixed_real_len"],
                    dtype=tf.float32)

            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer, config.train.hidden_layer_dropout_keep_prob)
            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"], data_processor.label_dict_file)

        super(TextRNNEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        assert config.TextRNN.cell_type in ModelHelper.VALID_CELL_TYPE
        # TextRNN only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
