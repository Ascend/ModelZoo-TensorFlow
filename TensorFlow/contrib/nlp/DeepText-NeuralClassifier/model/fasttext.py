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

from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper


class FastTextEstimator(NPUEstimator):
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        logger = data_processor.logger
        embedding_layer = EmbeddingLayer(config, logger=logger)
        model_helper = ModelHelper(config, logger=logger)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor)
            input_layer = []
            len_list = []
            for feature_name in params["feature_names"]:
                index = data_processor.dict_names.index(feature_name)
                input_layer.append(embedding_layer.get_vocab_embedding_sparse(
                    feature_name, features["var_len_" + feature_name],
                    len(data_processor.dict_list[index]), params["epoch"],
                    pretrained_embedding_file=
                    data_processor.pretrained_embedding_files[index],
                    dict_map=data_processor.dict_list[index],
                    mode=mode))
                len_list.append(features[feature_name + "_var_real_len"])
                if data_processor.ngram_list[index] > 1:
                    ngram_name = feature_name + "_ngram"
                    index = data_processor.dict_names.index(ngram_name)
                    input_layer.append(
                        embedding_layer.get_vocab_embedding_sparse(
                            ngram_name, features["var_len_" + ngram_name],
                            len(data_processor.dict_list[index]),
                            params["epoch"],
                            mode=mode))
                    len_list.append(features[ngram_name + "_var_real_len"])
            hidden_layer = input_layer[0]
            total_len = len_list[0]
            for i in range(1, len(input_layer)):
                hidden_layer = hidden_layer + input_layer[i]
                total_len = total_len + len_list[i]
            hidden_layer = tf.div(hidden_layer, total_len)
            hidden_layer = tf.contrib.layers.fully_connected(
            inputs=hidden_layer, num_outputs=256, activation_fn=tf.nn.relu)
            hidden_layer = tf.contrib.layers.fully_connected(
            inputs=hidden_layer, num_outputs=config.embedding_layer.embedding_dimension, activation_fn=tf.nn.relu)

            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer, config.train.hidden_layer_dropout_keep_prob)
            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"], data_processor.label_dict_file)

        super(FastTextEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)
        super(FastTextEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor):
        for feature_name in feature_names:
            assert feature_name in data_processor.dict_names
            index = data_processor.dict_names.index(feature_name)
            assert len(data_processor.dict_list[index]) > 0
