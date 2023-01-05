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


class TextVDCNNEstimator(NPUEstimator):
    VALID_DEPTH = [9, 17, 29, 49]

    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _convolutional_block(inputs, num_layers, num_filters, name, mode):
            """Convolutional Block of VDCNN
            Convolutional block contains 2 conv layers, and can be repeated
            Temp Conv-->Batch Norm-->ReLU-->Temp Conv-->Batch Norm-->ReLU
            """
            with tf.variable_scope("conv_block_%s" % name):
                is_training = False
                if mode == tf.estimator.ModeKeys.TRAIN:
                    is_training = True
                hidden_layer = inputs
                initializer_normal = tf.random_normal_initializer(stddev=0.1)
                initializer_const = tf.constant_initializer(0.0)
                for i in range(0, num_layers):
                    filter_shape = [3, 1, hidden_layer.get_shape()[3],
                                    num_filters]
                    w = tf.compat.v1.get_variable(name='W_' + str(i), shape=filter_shape,
                                        initializer=initializer_normal)
                    b = tf.compat.v1.get_variable(name='b_' + str(i), shape=[num_filters],
                                        initializer=initializer_const)
                    conv = tf.nn.conv2d(hidden_layer, w, strides=[1, 1, 1, 1],
                                        padding="SAME")
                    conv = tf.nn.bias_add(conv, b)
                    batch_norm = tf.layers.batch_normalization(
                        conv, center=True, scale=True, training=is_training)
                    hidden_layer = tf.nn.relu(batch_norm)
                return hidden_layer

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)

            """VDCNN architecture
              1. text(char is recommended)
              2. embedding lookup
              3. conv layer(64 feature maps)
              4. conv blocks(contains 2 conv layers, and can be repeated)
              5. fc1
              6. fc2
              7. fc3(softmax)
              pooling is importmant and shortcut is optional
            """
            sequence_length = data_processor.max_sequence_length[index]
            # embedding shape [batch_size, sequence_length, embedding_dimension]
            embedding = embedding_layer.get_vocab_embedding(
                feature_name, features["fixed_len_" + feature_name],
                len(data_processor.dict_list[index]), params["epoch"],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index],
                mode=mode)
            embedding = tf.reshape(embedding,
                                   [-1, sequence_length,
                                    config.embedding_layer.embedding_dimension])
            embedding = tf.expand_dims(embedding, -1)
            if mode == tf.estimator.ModeKeys.TRAIN:
                embedding = model_helper.dropout(
                    embedding,
                    config.embedding_layer.embedding_dropout_keep_prob)

            initializer = tf.random_normal_initializer(stddev=0.1)
            # first conv layer (filter_size=3, #feature_map=64)
            with tf.variable_scope("first_conv") as scope:
                filter_shape = [3, config.embedding_layer.embedding_dimension,
                                1, 64]
                w = tf.compat.v1.get_variable(name='W_1', shape=filter_shape,
                                    initializer=initializer)
                """
                  argv1: input = [batch_size, in_height, in_width, in_channels]
                  argv2: filter = [filter_height, filter_width, in_channels,
                                   out_channels]
                  argv3: strides
                  return: feature_map
                  note:
                    1. out_channels = num_filters = #feature map
                    2. for padding="SAME", new_height=new_width=
                           ceil(input_size/stride)
                       for padding="VALID", new_height=new_width=
                           ceil((input_size-filter_size+1)/stride)
                """
                conv = tf.nn.conv2d(
                    embedding, w,
                    strides=[1, 1, config.embedding_layer.embedding_dimension,
                             1],
                    padding="SAME")
                b = tf.compat.v1.get_variable(name='b_1', shape=[64],
                                    initializer=tf.constant_initializer(0.0))
                out = tf.nn.bias_add(conv, b)
                first_conv = tf.nn.relu(out)

            """all convolutional blocks
            4 kinds of conv blocks, which #feature_map are 64,128,256,512
            Depth:             9  17 29 49
            ------------------------------
            conv block 512:    2  4  4  6
            conv block 256:    2  4  4  10
            conv block 128:    2  4  10 16
            conv block 64:     2  4  10 16
            First conv. layer: 1  1  1  1
            """
            vdcnn_depth = {}
            vdcnn_depth[9] = [2, 2, 2, 2]
            vdcnn_depth[17] = [4, 4, 4, 4]
            vdcnn_depth[29] = [10, 10, 4, 4]
            vdcnn_depth[49] = [16, 16, 10, 6]
            max_pool_ksize = [1, 3, 1, 1]
            max_pool_strides = [1, 2, 1, 1]
            num_filters = [64, 128, 256, 512]
            conv_block = first_conv
            for i in range(0, 4):
                conv_block = _convolutional_block(
                    conv_block,
                    num_layers=vdcnn_depth[config.TextVDCNN.vdcnn_depth][i],
                    num_filters=num_filters[i], name="cb_" + str(i), mode=mode)
                pool = tf.nn.max_pool2d(conv_block, ksize=max_pool_ksize,
                                      strides=max_pool_strides,
                                      padding='SAME', name="pool_" + str(i))

            pool_shape = int(np.prod(pool.get_shape()[1:]))
            pool = tf.reshape(pool, (-1, pool_shape))

            # fc1
            fc1 = tf.contrib.layers.fully_connected(
                inputs=pool, num_outputs=2048,
                activation_fn=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                fc1 = model_helper.dropout(
                    fc1, config.train.hidden_layer_dropout_keep_prob)
            # fc2
            hidden_layer = tf.contrib.layers.fully_connected(
                inputs=fc1, num_outputs=2048,
                activation_fn=tf.nn.relu)
            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer, config.train.hidden_layer_dropout_keep_prob)
            # fc3(softmax)
            return model_helper.get_softmax_estimator_spec(
                hidden_layer, mode, labels, params["label_size"],
                params["static_embedding"])

        super(TextVDCNNEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        # TextVDCNN only support one feature and should be char or token.
        # char input is recommended.
        assert len(feature_names) == 1
        assert config.TextVDCNN.vdcnn_depth in TextVDCNNEstimator.VALID_DEPTH
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
