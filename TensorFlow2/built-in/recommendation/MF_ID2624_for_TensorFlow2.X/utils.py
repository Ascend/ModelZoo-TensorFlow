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
Created on Mar 26, 2020

Matrix Factorization

All ratings are contained in the file "ratings.dat" and are in the
following format:
UserID::MovieID::Rating::Timestamp
- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

@author: Gengziyao
"""
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time

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


def create_explicit_ml_1m_dataset(file, latent_dim=4, test_size=0.2):
    """
    create the explicit dataset of movielens-1m
    We took the last 20% of each user sorted by timestamp as the test dataset
    Each of these samples contains UserId, MovieId, Rating, avg_score
    :param file: dataset path
    :param latent_dim: latent factor
    :param test_size: ratio of test dataset
    :return: user_num, item_num, train_df, test_df
    """
    data_df = pd.read_csv(file, sep="::", engine='python',
                     names=['UserId', 'MovieId', 'Rating', 'Timestamp'])
    data_df['avg_score'] = data_df.groupby(by='UserId')['Rating'].transform('mean')
    # feature columns
    user_num, item_num = data_df['UserId'].max() + 1, data_df['MovieId'].max() + 1
    feature_columns = [[denseFeature('avg_score')],
                       [sparseFeature('user_id', user_num, latent_dim),
                        sparseFeature('item_id', item_num, latent_dim)]]
    # split train dataset and test dataset
    watch_count = data_df.groupby(by='UserId')['MovieId'].agg('count')
    test_df = pd.concat([
        data_df[data_df.UserId == i].iloc[int(0.8 * watch_count[i]):] for i in tqdm(watch_count.index)], axis=0)
    test_df = test_df.reset_index()
    train_df = data_df.drop(labels=test_df['index'])
    train_df = train_df.drop(['Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)
    test_df = test_df.drop(['index', 'Timestamp'], axis=1).sample(frac=1.).reset_index(drop=True)

    train_X = [train_df['avg_score'].values, train_df[['UserId', 'MovieId']].values.astype('int32')]
    train_y = train_df['Rating'].values.astype('float32')
    test_X = [test_df['avg_score'].values, test_df[['UserId', 'MovieId']].values.astype('int32')]
    test_y = test_df['Rating'].values.astype('float32')
    return feature_columns, (train_X, train_y), (test_X, test_y)

class LossHistory(tf.keras.callbacks.Callback):
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        print('step:%d ,loss: %f ,time:%f'%(batch, loss, time.time() - self.start), flush=True)
    def on_epoch_begin(self, epoch, logs={}):
        self.epochstart = time.time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time.time() - self.epochstart
        print('epoch_duration: ', duration)
        if epoch != 0:
            self.perf.append(duration)
    def on_train_begin(self, logs={}):
        self.batch_size = 797696 // self.params['steps']
        self.samples = self.batch_size * self.params['steps']
        print('params: ', self.params)
        self.perf = []
    def on_train_end(self, logs={}):
        print('imgs/s: %.2f'%(self.samples / np.mean(self.perf)))
# create_explicit_ml_1m_dataset('../dataset/ml-1m/movies.dat')
