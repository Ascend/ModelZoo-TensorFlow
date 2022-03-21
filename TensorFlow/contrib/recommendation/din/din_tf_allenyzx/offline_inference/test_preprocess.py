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


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    test_data = []

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
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # create dataframe
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')


    pkl_save('./raw_data/cache.pkl', {'test_X': test_X, 'test_y': test_y, 'feature_columns': feature_columns})

    return test_X, test_y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='../raw_data/remap.pkl')
    parser.add_argument('--embed_dim', default=8,type=int)
    parser.add_argument('--maxlen', default=40,type=int)
    args = parser.parse_args()
    create_amazon_electronic_dataset(file=args.file, embed_dim=args.embed_dim, maxlen=args.maxlen)
