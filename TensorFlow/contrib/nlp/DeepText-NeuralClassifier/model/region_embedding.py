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


class RegionEmbeddingEstimator(NPUEstimator):
    VALID_MODE = ["WC", "CW"]

    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)
            region_radius = int(config.RegionEmbedding.region_size / 2)
            sequence_length = data_processor.max_sequence_length[index] + \
                              region_radius * 2
            vocab_ids = features["fixed_len_" + feature_name]
            padding_id = \
                data_processor.dict_list[index][data_processor.VOCAB_PADDING]
            vocab_ids = tf.pad(vocab_ids, tf.constant(
                [[0, 0], [region_radius, region_radius]]),
                               constant_values=padding_id)
            region_emb = embedding_layer.get_region_embedding(
                feature_name,
                vocab_ids,
                len(data_processor.dict_list[index]),
                params["epoch"],
                sequence_length,
                config.RegionEmbedding.region_size,
                config.RegionEmbedding.region_embedding_mode,
                mode,
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index])

            # which words have corresponding region embedding
            trimmed_seq = \
                vocab_ids[..., region_radius: sequence_length - region_radius]

            def mask(x):
                return tf.cast(tf.not_equal(tf.cast(x, tf.int32),
                                            tf.constant(padding_id)),
                               tf.float32)

            # remove padding(setting to zero)
            weight = tf.map_fn(mask, trimmed_seq, dtype=tf.float32,
                               back_prop=False)
            weight = tf.expand_dims(weight, -1)
            weighted_emb = region_emb * weight
            # document embedding
            hidden_layer = tf.reduce_sum(weighted_emb, 1)

            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"], data_processor.label_dict_file)

        super(RegionEmbeddingEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        # Region Embedding only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert config.RegionEmbedding.region_embedding_mode in \
            RegionEmbeddingEstimator.VALID_MODE
        assert config.RegionEmbedding.region_size % 2 == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
