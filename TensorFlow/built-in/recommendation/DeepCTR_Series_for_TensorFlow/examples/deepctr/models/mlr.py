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
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""
from npu_bridge.npu_init import *
from tensorflow.python.keras.layers import Activation, dot
from tensorflow.python.keras.models import Model

from ..feature_column import build_input_features, get_linear_logit
from ..layers.core import PredictionLayer
from ..layers.utils import concat_func


def MLR(region_feature_columns, base_feature_columns=None, region_num=4,
        l2_reg_linear=1e-5, seed=1024, task='binary',
        bias_feature_columns=None):
    """Instantiates the Mixed Logistic Regression/Piece-wise Linear Model.

    :param region_feature_columns: An iterable containing all the features used by region part of the model.
    :param base_feature_columns: An iterable containing all the features used by base part of the model.
    :param region_num: integer > 1,indicate the piece number
    :param l2_reg_linear: float. L2 regularizer strength applied to weight
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param bias_feature_columns: An iterable containing all the features used by bias part of the model.
    :return: A Keras model instance.
    """

    if region_num <= 1:
        raise ValueError("region_num must > 1")

    if base_feature_columns is None or len(base_feature_columns) == 0:
        base_feature_columns = region_feature_columns

    if bias_feature_columns is None:
        bias_feature_columns = []

    features = build_input_features(region_feature_columns + base_feature_columns + bias_feature_columns)

    inputs_list = list(features.values())

    region_score = get_region_score(features, region_feature_columns, region_num, l2_reg_linear, seed)
    learner_score = get_learner_score(features, base_feature_columns, region_num, l2_reg_linear, seed, task=task)

    final_logit = dot([region_score, learner_score], axes=-1)

    if bias_feature_columns is not None and len(bias_feature_columns) > 0:
        bias_score = get_learner_score(features, bias_feature_columns, 1, l2_reg_linear, seed, prefix='bias_',
                                       task='binary')

        final_logit = dot([final_logit, bias_score], axes=-1)

    model = Model(inputs=inputs_list, outputs=final_logit)
    return model


def get_region_score(features, feature_columns, region_number, l2_reg, seed, prefix='region_', seq_mask_zero=True):
    region_logit = concat_func([get_linear_logit(features, feature_columns, seed=seed + i,
                                                 prefix=prefix + str(i + 1), l2_reg=l2_reg) for i in
                                range(region_number)])
    return Activation('softmax')(region_logit)


def get_learner_score(features, feature_columns, region_number, l2_reg, seed, prefix='learner_', seq_mask_zero=True,
                      task='binary'):
    region_score = [PredictionLayer(task=task, use_bias=False)(
        get_linear_logit(features, feature_columns, seed=seed + i, prefix=prefix + str(i + 1),
                         l2_reg=l2_reg)) for i in
        range(region_number)]

    return concat_func(region_score)

