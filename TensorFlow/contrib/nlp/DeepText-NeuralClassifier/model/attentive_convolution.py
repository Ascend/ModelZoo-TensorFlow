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



class AttentiveConvNetEstimator(NPUEstimator):
    VALID_VERSION = ["light", "advanced"]
    def __init__(self, data_processor, model_params):
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            self._check(params["feature_names"], data_processor, config)
            feature_name = params["feature_names"][0]
            index = data_processor.dict_names.index(feature_name)
            sequence_length = data_processor.max_sequence_length[index]
            embedding = embedding_layer.get_vocab_embedding(
                feature_name, features["fixed_len_" + feature_name],
                len(data_processor.dict_list[index]), params["epoch"],
                pretrained_embedding_file=
                data_processor.pretrained_embedding_files[index],
                dict_map=data_processor.dict_list[index],
                mode=mode)
            dimension = config.embedding_layer.embedding_dimension
            hidden_size = config.AttentiveConvNet.attentive_hidden_size

            # first fully connected matrix 
            mat_hidden1 = tf.compat.v1.get_variable("mat_hidden1",
                    shape=[dimension, hidden_size],
                    initializer=tf.random.uniform_initializer(
                    -1.0 * pow(6.0 / (dimension + hidden_size), 0.5),
                    pow(6.0 / (hidden_size + dimension), 0.5)))
            bias_hidden1 = tf.compat.v1.get_variable("bias_hidden1", shape=[hidden_size])
            # second fully connected matrix 
            mat_hidden2 = tf.compat.v1.get_variable("mat_hidden2",
                    shape=[hidden_size, hidden_size],
                    initializer=tf.random.uniform_initializer(
                    -1.0 * pow(3.0 / hidden_size, 0.5),
                    pow(3.0 / hidden_size, 0.5)))
            bias_hidden2 = tf.compat.v1.get_variable("bias_hidden2", shape=[hidden_size])

            def _gconv(context, filter_width, name):
                """ compute equations 7,8,9
                """
                bias_ha = tf.compat.v1.get_variable(name + "_bias_ha", shape=[dimension])
                bias_ga = tf.compat.v1.get_variable(name + "_bias_ga", shape=[dimension])
                embedded_context = tf.expand_dims(context, -1)
                filter_shape = [filter_width, dimension, 1, dimension]
                filter_o = tf.Variable(
                        tf.truncated_normal(filter_shape, stddev=0.1),
                        name=name + "_filter_Wo")
                filter_g = tf.Variable(
                        tf.truncated_normal(filter_shape, stddev=0.1),
                        name=name + "_filter_Wg")
                conv_o = tf.nn.conv2d(embedded_context, filter_o,
                        strides=[1, 1, dimension, 1], padding="SAME",
                        name=name + "_convolution_Wo")
                conv_g = tf.nn.conv2d(embedded_context, filter_g,
                        strides=[1, 1, dimension, 1], padding="SAME",
                        name=name + "_convolution_Wg")
                conv_o = tf.keras.backend.permute_dimensions(conv_o,
                        (0, 1, 3, 2))
                conv_g = tf.keras.backend.permute_dimensions(conv_g,
                        (0, 1, 3, 2))
                o_context = tf.tanh(tf.nn.bias_add(tf.squeeze(conv_o, [-1]),
                    bias_ha), name=name + "_Wo_tanh")
                g_context = tf.sigmoid(tf.nn.bias_add(tf.squeeze(conv_g, [-1]),
                    bias_ga), name=name + "_Wg_sigmoid")
                return g_context * context + (1 - g_context) * o_context

            def _attentive_context(source, focus, name="context_generate"):
                if config.AttentiveConvNet.attentive_version == 'advanced':
                    mat_dimension = 2 * dimension
                else:
                    mat_dimension = dimension
                mat_tx = tf.compat.v1.get_variable(name + "mat_tx", shape=[mat_dimension,
                    mat_dimension], initializer=tf.random.uniform_initializer(
                        -1.0 * pow(3.0 / dimension, 0.5),
                        pow(3.0 / dimension, 0.5)))
                mat_ta = tf.compat.v1.get_variable(name + "mat_ta", shape=[dimension,
                    mat_dimension], initializer=tf.random.uniform_initializer(
                        -1.0 * pow(3.0 / dimension, 0.5),
                        pow(3.0 / dimension, 0.5)))
                # use dot and batch_dot in keras, compute equation 2
                embedding_conv = tf.keras.backend.dot(source, mat_tx)
                scores = tf.keras.backend.batch_dot(embedding_conv,
                        tf.keras.backend.permute_dimensions(focus, (0, 2, 1)))
                scores_softmax = tf.keras.activations.softmax(scores, axis=1)
                # computes the context featur_map like equation 4
                res = tf.matmul(scores_softmax, focus)
                # weights the output for equation 6
                context = tf.keras.backend.permute_dimensions(
                        tf.keras.backend.dot(mat_ta,
                            tf.keras.backend.permute_dimensions(res,
                                (0, 2, 1))), (1, 2, 0))
                return context

            def _attentive_convolution(benificiary, attentive_context,
                                      name="attentive_convolution"):
                """ compute equation 6
                """
                bias = tf.compat.v1.get_variable(name + "bias", shape=[dimension])
                embedded_text = tf.expand_dims(benificiary, -1)
                filter_shape = [config.AttentiveConvNet.attentive_width,
                                dimension, 1, dimension]
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape,
                    stddev=0.1), name=name + "filter")
                convolution = tf.nn.conv2d(embedded_text, conv_filter,
                    strides=[1, 1, dimension, 1], padding="SAME",
                    name=name + "convolutioin")
                convolution = tf.keras.backend.permute_dimensions(
                    convolution, (0, 1, 3, 2))
                conv_text = tf.squeeze(convolution, [-1])
                merge_text = tf.add(attentive_context, conv_text)
                merge_text = tf.nn.bias_add(merge_text, bias)
                tanh_out = tf.tanh(merge_text, name=name + "tanh")
                tanh_out = tf.expand_dims(tanh_out, -1)

                return tanh_out

            if config.AttentiveConvNet.attentive_version == "advanced":
                # generate source
                source_x_uni = _gconv(embedding, 1, "source_uni")
                source_x_tri = _gconv(embedding, 3, "source_tri")
                x_mgran = tf.concat([source_x_uni, source_x_tri], -1)
                # generate focus
                focus_a_uni = _gconv(embedding, 1, "focus_uni")
                focus_a_tri = _gconv(embedding, 3, "focus_tri")
                a_mgran = tf.concat([focus_a_uni, focus_a_tri], -1)
                # generate benificiary
                x_benificiary = _gconv(embedding, 1, "beni_uni")
            else:
                # light version
                x_mgran, a_mgran, x_benificiary = \
                        embedding, embedding, embedding

            context = _attentive_context(x_mgran, a_mgran)
            attentive_embedding = _attentive_convolution(x_benificiary, context)
            pooled = tf.nn.max_pool2d(attentive_embedding,
                ksize=[1, sequence_length, 1, 1], strides=[1, 1, 1, 1],
                padding="VALID", name="max_pooling")
            hidden_layer = tf.reshape(
                pooled, [-1, config.embedding_layer.embedding_dimension])

            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer = model_helper.dropout(
                    hidden_layer, config.train.hidden_layer_dropout_keep_prob)
            hidden_layer1 = tf.nn.relu(tf.matmul(hidden_layer, mat_hidden1) +
                    bias_hidden1, name="relu_hidden1")
            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer1 = model_helper.dropout(
                    hidden_layer1, config.train.hidden_layer_dropout_keep_prob)
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, mat_hidden2) +
                    bias_hidden2, name="relu_hidden2")
            if mode == tf.estimator.ModeKeys.TRAIN:
                hidden_layer2 = model_helper.dropout(
                    hidden_layer2, config.train.hidden_layer_dropout_keep_prob)
            # concat max pooling, hidden layer 1 output, hidden layer 2 output
            output = tf.concat([hidden_layer, hidden_layer1, hidden_layer2], -1)

            return model_helper.get_softmax_estimator_spec(
                output, mode, labels, params["label_size"],
                params["static_embedding"])

        super(AttentiveConvNetEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)

    @staticmethod
    def _check(feature_names, data_processor, config):
        # AttentiveConvNet only support one feature and should be char or token.
        assert len(feature_names) == 1
        assert config.AttentiveConvNet.attentive_version in \
               AttentiveConvNetEstimator.VALID_VERSION
        assert config.AttentiveConvNet.attentive_width % 2 == 1
        assert feature_names[0] in data_processor.dict_names
        index = data_processor.dict_names.index(feature_names[0])
        assert len(data_processor.dict_list[index]) > 0
        assert data_processor.max_sequence_length[index] > 0
