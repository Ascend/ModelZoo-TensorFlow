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
Created on Nov 13, 2020

evaluate model

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np


def getHit(df):
    """
    calculate hit rate
    :return:
    """
    if sum(df['pred']) < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    """
    calculate NDCG
    :return:
    """
    if sum(df['pred']) < _K:
        return 1 / np.log(sum(df['pred']) + 2)
    else:
        return 0.


def getMRR(df):
    """
    calculate MRR
    :return:
    """
    return 1 / (sum(df['pred']) + 1)


def evaluate_model(model, test, K):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    global _K
    _K = K
    test_X = test
    # predict
    pos_score, neg_score = model.predict(test_X)
    # create dataframe
    test_df = pd.DataFrame(test_X[0], columns=['user_id'])
    # if mode == 'inner', pos score < neg score, pred = 1
    if model.mode == 'inner':
        test_df['pred'] = (pos_score <= neg_score).astype(np.int32)
    else:
        test_df['pred'] = (pos_score >= neg_score).astype(np.int32)
    # groupby
    tg = test_df.groupby('user_id')
    # calculate hit
    hit_rate = tg.apply(getHit).mean()
    # calculate ndcg
    ndcg = tg.apply(getNDCG).mean()
    # calculate mrr
    mrr = tg.apply(getMRR).mean()
    return hit_rate, ndcg, mrr