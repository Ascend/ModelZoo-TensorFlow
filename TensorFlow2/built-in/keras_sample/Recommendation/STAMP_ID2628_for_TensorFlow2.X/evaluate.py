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
'''
Descripttion: Evaluate
Author: Ziyao Geng
Date: 2020-10-25 10:07:17
LastEditors: ZiyaoGeng
LastEditTime: 2020-10-26 12:47:28
'''

import numpy as np


def getHit(pred_y, true_y):
    """
    calculate hit rate
    :return:
    """
    # reversed
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([true_y[i] in pred_index[i] for i in range(len(pred_index))]) / len(pred_index)


def getMRR(pred_y, true_y):
    """
    """
    pred_index = np.argsort(-pred_y)[:, :_K]
    return sum([1 / (np.where(true_y[i] == pred_index[i])[0][0] + 1) \
        for i in range(len(pred_index)) if len(np.where(true_y[i] == pred_index[i])[0]) != 0]) / len(pred_index)


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, mrr
    """
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    hit_rate = getHit(pred_y, test_y)
    mrr = getMRR(pred_y, test_y)
    
    
    return hit_rate, mrr