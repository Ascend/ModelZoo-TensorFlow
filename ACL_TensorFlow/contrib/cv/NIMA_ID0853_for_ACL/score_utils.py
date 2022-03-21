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

import numpy as np
from scipy.stats import spearmanr
# calculate mean score for AVA dataset
# def mean_score(scores):
#     si = np.arange(1, 11, 1)
#     print("--------------------",(scores*si).shape)
#     mean = np.sum(scores * si)
#     return mean

# # calculate standard deviation of scores for AVA dataset
# def std_score(scores):
#     si = np.arange(1, 11, 1)
#     mean = mean_score(scores)
#     std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
#     return std


def mean_score(scores):
    """ calculate mean score for AVA dataset
    :param scores:
    :return: row wise mean score if scores contains multiple rows, else
             a single mean score
    """
    si = np.arange(1, 11, 1).reshape(1,10)
    mean = np.sum(scores * si, axis=1)
    if mean.shape==(1,):
        mean = mean[0]
    return mean


def std_score(scores):
    """ calculate standard deviation of scores for AVA dataset
    :param scores:
    :return: row wise standard deviations if scores contains multiple rows,
             else a single standard deviation
    """
    si = np.arange(1, 11, 1).reshape(1,10)
    mean = mean_score(scores).reshape(-1,1)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores, axis=1))
    if std.shape==(1,):
        std = std[0]
    return std



def srcc(y_true, y_pred):
    """ calculate spearman's rank correlation coefficient
    :param y_test: the human ratings (width 10)
    :param y_pred: the predicted ratings (width 10)
    :return:
    """
    # print("--------------y_true----------------",y_true.shape)
    # print("--------------y_pred----------------",y_pred.shape)
    
    # print("mean---score----",mean_score(y_true))
    # print("mean---score----",mean_score(y_pred))
    rho, pValue = spearmanr(mean_score(y_true), mean_score(y_pred))
    # print(rho)
    return rho