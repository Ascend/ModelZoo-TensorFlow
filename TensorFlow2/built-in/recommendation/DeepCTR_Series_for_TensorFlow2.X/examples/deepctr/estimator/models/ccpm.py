# -*- coding:utf-8 -*-
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
"""

Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
    (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)

"""
#from npu_bridge.npu_init import *
import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
from ...layers.core import DNN
from ...layers.sequence import KMaxPooling
from ...layers.utils import concat_func


def CCPMEstimator(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(6, 5), conv_filters=(4, 4),
                  dnn_hidden_units=(128, 64), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
                  seed=1024, task='binary', model_dir=None, config=None, linear_optimizer='Ftrl',
                  dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the Convolutional Click Prediction Model architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
    :param config: tf.RunConfig object to configure the runtime settings.
    :param linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. Defaults to FTRL optimizer.
    :param dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. Defaults to Adagrad optimizer.
    :param training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run on the chief worker during training.
    :return: A Tensorflow Estimator  instance.

    """

    if len(conv_kernel_width) != len(conv_filters):
        raise ValueError(
            "conv_kernel_width must have same element with conv_filters")

    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)

        linear_logits = get_linear_logit(features, linear_feature_columns, l2_reg_linear=l2_reg_linear)

        with variable_scope(DNN_SCOPE_NAME):
            sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 l2_reg_embedding=l2_reg_embedding)
            n = len(sparse_embedding_list)
            l = len(conv_filters)

            conv_input = concat_func(sparse_embedding_list, axis=1)
            pooling_result = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(x, axis=3))(conv_input)

            for i in range(1, l + 1):
                filters = conv_filters[i - 1]
                width = conv_kernel_width[i - 1]
                k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3

                conv_result = tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1),
                                                     padding='same',
                                                     activation='tanh', use_bias=True, )(pooling_result)
                pooling_result = KMaxPooling(
                    k=min(k, int(conv_result.shape[1])), axis=1)(conv_result)

            flatten_result = tf.keras.layers.Flatten()(pooling_result)
            dnn_out = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, seed=seed)(flatten_result, training=train_flag)
            dnn_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

        logits = linear_logits + dnn_logit

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks
                                )

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=npu_run_config_init(run_config=config))

