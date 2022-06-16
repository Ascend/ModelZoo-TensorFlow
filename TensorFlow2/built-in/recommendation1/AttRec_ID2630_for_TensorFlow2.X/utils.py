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
Created on Nov 10, 2020

create implicit ml-1m dataset(update, delete dense_inputs, sparse_inputs)

This dataset is for AttRec model use.

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40):
    """
    :param file: A string. dataset path.
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param embed_dim: A scalar. latent factor.
    :param maxlen: A scalar. maxlen.
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    data_df = pd.read_csv(file, sep="::", engine='python',
                     names=['user_id', 'item_id', 'label', 'Timestamp'])
    # implicit dataset
    data_df = data_df[data_df.label >= trans_score]

    # sort
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    train_data, val_data, test_data = [], [], []

    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + 100)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                for neg in neg_list[i:]:
                    test_data.append([user_id, hist_i, pos_list[i], neg])
            elif i == len(pos_list) - 2:
                val_data.append([user_id, hist_i, pos_list[i], neg_list[i]])
            else:
                train_data.append([user_id, hist_i, pos_list[i], neg_list[i]])

    # feature columns
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    feature_columns = [sparseFeature('user_id', user_num, embed_dim),
                       sparseFeature('item_id', item_num, embed_dim)]

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'pos_item', 'neg_item'])
    print('==================Padding===================')

    # create dataset
    def df_to_list(data):
        return [data['user_id'].values, pad_sequences(data['hist'], maxlen=maxlen),
                data['pos_item'].values, data['neg_item'].values]

    train_X = df_to_list(train)
    val_X = df_to_list(val)
    test_X = df_to_list(test)
    print('============Data Preprocess End=============')
    return feature_columns, train_X, val_X, test_X


# create_implicit_ml_1m_dataset('../dataset/ml-1m/ratings.dat', maxlen=5)