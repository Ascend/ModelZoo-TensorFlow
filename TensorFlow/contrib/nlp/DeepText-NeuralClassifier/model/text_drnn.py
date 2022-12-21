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

from model import model_layer
from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper


class TextDRNNEstimator(NPUEstimator):
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)
            padding_id = \
                data_processor.dict_list[index][data_processor.VOCAB_PADDING]
            window_size = config.TextDRNN.drnn_window_size
            vocab_ids = tf.pad(
                features["fixed_len_" + feature_name],
                tf.constant([[0, 0], [window_size - 1, 0]]),
                constant_values=padding_id)
            embedding_lookup_table = embedding_layer.get_lookup_table(
                feature_name, len(data_processor.dict_list[index]),
                config.embedding_layer.embedding_dimension, params["epoch"],
                dict_map=data_processor.dict_list[index],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index], mode=mode)
            sequence_length = \
                data_processor.max_sequence_length[index] + window_size - 1
            aligned_seq = \
                [tf.slice(vocab_ids, [0, i], [-1, window_size])
                 for i in range(0, sequence_length - window_size + 1)]
            aligned_seq = \
                tf.reshape(tf.concat(list(aligned_seq), 1),
                           [-1, sequence_length - window_size + 1, window_size])
            embedding = tf.nn.embedding_lookup(embedding_lookup_table,
                                               aligned_seq)
            if mode == tf.estimator.ModeKeys.TRAIN:
                embedding = model_helper.dropout(
                    embedding,
                    config.embedding_layer.embedding_dropout_keep_prob)
            embedding = tf.reshape(
                embedding,
                [-1, window_size, config.embedding_layer.embedding_dimension])
            _, state = model_layer.recurrent(
                embedding, config.TextDRNN.drnn_rnn_dimension,
                cell_type=config.TextDRNN.drnn_cell_type,
                cell_hidden_keep_prob=
                config.TextDRNN.drnn_cell_hidden_keep_prob, mode=mode,
                use_bidirectional=False, name="drnn", reuse=None)
            state = tf.reshape(
                state, [-1, sequence_length - window_size + 1,
                        config.TextDRNN.drnn_rnn_dimension])

            if mode == tf.estimator.ModeKeys.TRAIN:
                state = model_layer.batch_norm(
                    state, tf.constant(True, dtype=tf.bool), name="bn")
            else:
                state = model_layer.batch_norm(
                    state, tf.constant(False, dtype=tf.bool), name="bn")
            state = tf.contrib.layers.fully_connected(
                state, config.embedding_layer.embedding_dimension,
                biases_initializer=None)

            def _mask_no_padding(x):
                return tf.cast(tf.not_equal(tf.cast(x, tf.int32),
                                            tf.constant(padding_id)),
                               tf.float32)

            def _mask_padding(x):
                return tf.cast(tf.equal(tf.cast(x, tf.int32),
                                        tf.constant(padding_id)),
                               tf.float32)

            trim_seq = vocab_ids[..., window_size - 1:]
            weight = tf.map_fn(_mask_no_padding, trim_seq, dtype=tf.float32,
                               back_prop=False)
            weight = tf.expand_dims(weight, -1)
            weighted_emb = state * weight
            neg = tf.map_fn(_mask_padding, trim_seq, dtype=tf.float32,
                            back_prop=False)
            neg = tf.expand_dims(neg, -1) * tf.float32.min
            weighted_emb = weighted_emb + neg
            hidden_layer = tf.reduce_max(weighted_emb, axis=1)

            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer,
                    config.train.hidden_layer_dropout_keep_prob)
            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"],
                data_processor.label_dict_file)

        super(TextDRNNEstimator, self).__init__(
            model_fn=_model_fn,
            model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        assert config.TextDRNN.drnn_cell_type in ModelHelper.VALID_CELL_TYPE
        # TextDRNN only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert data_processor.dict_list[index]
        assert data_processor.max_sequence_length[index] > 0
