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
# Copyright 2020 Huawei Technologies Co., Ltd
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
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
random.seed(1234)


def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file)
    pkl_file.close()
    return file


def pkl_save(filename, model):
    output = open(filename, 'wb')
    pickle.dump(model, output)
    output.close()


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=100,cache=False):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    if cache and os.path.exists('raw_data/cache.pkl'):
        res = pkl_load('raw_data/cache.pkl')
        feature_columns = res['feature_columns']
        behavior_list = res['behavior_list']
        train_X = res['train_X']
        train_y = res['train_y']
        val_X = res['val_X']
        val_y = res['val_y']
        return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y)

    print('==========Data Preprocess Start============')
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data = [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i - 1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                pass
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               # np.array(train['hist'].tolist()),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             # np.array(val['hist'].tolist()),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    print('============Data Preprocess End=============')

    pkl_save('raw_data/cache.pkl', {'feature_columns': feature_columns,
                                    'behavior_list': behavior_list,
                                    'train_X': train_X,
                                    'train_y': train_y,
                                    'val_X': val_X,
                                    'val_y': val_y})

    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y)


class DatasetIterater(object):
    def __init__(self, batches, batch_size, batches_len=None):
        self.batch_size = batch_size
        self.batches = batches
        if batches_len is None:
            batches_len = len(batches)
        self.n_batches = batches_len // batch_size
        self.index = 0

    def _to_tensor(self, batches, si, ei):
        train_X, train_y = batches
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = train_X
        sparse_inputs = sparse_inputs[si:ei]
        seq_inputs = seq_inputs[si:ei]
        item_inputs = item_inputs[si:ei]
        train_y = train_y[si:ei]
        return dense_inputs, sparse_inputs, seq_inputs, item_inputs, train_y

    def shuffle(self):
        print('========== shuffle ============')
        train_X, train_y = self.batches
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = train_X

        shuffle_index = np.array(range(len(train_y)))
        np.random.shuffle(shuffle_index)
        train_y = np.array([train_y[i] for i in shuffle_index])
        dense_inputs = np.array([dense_inputs[i] for i in shuffle_index])
        sparse_inputs = np.array([sparse_inputs[i] for i in shuffle_index])
        seq_inputs = np.array([seq_inputs[i] for i in shuffle_index])
        item_inputs = np.array([item_inputs[i] for i in shuffle_index])

        self.batches = (dense_inputs, sparse_inputs, seq_inputs, item_inputs), train_y
        self.index = 0

    def __next__(self):
        if self.index + self.batch_size >= self.n_batches:
            self.shuffle()
            return self._to_tensor(self.batches, 1, self.batch_size)
        else:
            self.index += 1
            batches = self._to_tensor(self.batches, self.index * self.batch_size, (self.index + 1) * self.batch_size)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches