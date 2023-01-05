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
import numpy as np
import tensorflow as tf

from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper


class TextDPCNNEstimator(NPUEstimator):
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _convolution(inputs, num_filters, name):
            """two layers of convolution
            """

            with tf.variable_scope("two_conv-%s" % name):
                initializer_normal = tf.random_normal_initializer(stddev=0.01)
                filter_shape = [3, 1, num_filters, num_filters]

                W1 = tf.compat.v1.get_variable(name="W1-%s" % name, shape=filter_shape,
                                     initializer=initializer_normal)
                b1 = tf.compat.v1.get_variable(name="b1-%s" % name, shape=[num_filters])
                # pre - activation, before convolution
                relu1 = tf.nn.relu(inputs, name="relu1-%s" % name)
                conv1 = tf.nn.conv2d(relu1, W1, strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     name="convolution1-%s" % name)
                conv1 = tf.nn.bias_add(conv1, b1)

                W2 = tf.compat.v1.get_variable(name="W2-%s" % name, shape=filter_shape,
                                     initializer=initializer_normal)
                b2 = tf.compat.v1.get_variable(name="b2-%s" % name, shape=[num_filters])
                # pre - activation
                relu2 = tf.nn.relu(conv1, name="relu2-%s" % name)
                conv2 = tf.nn.conv2d(relu2, W2, strides=[1, 1, 1, 1],
                                     padding="SAME",
                                     name="convolution2-%s" % name)
                conv2 = tf.nn.bias_add(conv2, b2)
            # return shortcut connections with identity mapping
            return inputs + conv2

        def _convolution_block(inputs, num_filters, name):
            """DPCNN Block architecture
              1. pooling (strides=2, sequence halved)
              2. relu
              3. conv1 layer
              4. relu
              5. conv2 layer
              6. return pooling output + conv2 layer output
            """

            with tf.variable_scope("pooling-%s" % name):
                pooled = tf.nn.max_pool2d(inputs, ksize=[1, 3, 1, 1],
                                        strides=[1, 2, 1, 1], padding='SAME',
                                        name="max-pooling-%s" % name)

            return _convolution(pooled, num_filters, name)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)

            num_filters = config.TextDPCNN.num_filters
            sequence_length = data_processor.max_sequence_length[index]
            embedding = embedding_layer.get_vocab_embedding(
                feature_name, features["fixed_len_" + feature_name],
                len(data_processor.dict_list[index]), params["epoch"],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index],
                mode=mode)
            embedding_dims = config.embedding_layer.embedding_dimension
            embedding = tf.reshape(embedding,
                                   [-1, sequence_length, embedding_dims])
            embedding = tf.expand_dims(embedding, -1)
            if mode == tf.estimator.ModeKeys.TRAIN:
                embedding = model_helper.dropout(
                    embedding,
                    config.embedding_layer.embedding_dropout_keep_prob)

            initializer = tf.random_normal_initializer(stddev=0.01)
            with tf.variable_scope("dpcnn") as scope:
                filter_shape = [3, embedding_dims, 1, num_filters]
                W0 = tf.compat.v1.get_variable(name="W0", shape=filter_shape,
                                     initializer=initializer)
                b0 = tf.compat.v1.get_variable(name="b0", shape=[num_filters])
                conv0 = tf.nn.conv2d(embedding, W0,
                                     strides=[1, 1, embedding_dims, 1],
                                     padding="SAME")
                conv0 = tf.nn.bias_add(conv0, b0)

                conv = _convolution(conv0, num_filters, "conv-1-2")

                for i in range(config.TextDPCNN.dpcnn_blocks):
                    conv = _convolution_block(conv, num_filters,
                                              "convolution-block-%d" % i)

                outputs_shape = int(np.prod(conv.get_shape()[1:]))
                outputs = tf.reshape(conv, (-1, outputs_shape))

            return model_helper.get_softmax_estimator_spec(
                outputs, mode, labels, params["label_size"],
                params["static_embedding"],data_processor.label_dict_file)

        super(TextDPCNNEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        # TextDPCNN only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
