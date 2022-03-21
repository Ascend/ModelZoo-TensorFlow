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
import random
import pickle
import numpy as np
import pandas as pd

random.seed(2020)


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


# reviews
reviews_df = pd.read_pickle('../raw_data/reviews.pkl')
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# meta
meta_df = pd.read_pickle('../raw_data/meta.pkl')
meta_df = meta_df[['asin', 'categories']]

meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# meta_df asin mapping
asin_map, asin_key = build_map(meta_df, 'asin')
# meta_df categories mapping
cate_map, cate_key = build_map(meta_df, 'categories')
# reviews_df user mapping
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

# user_count: 192403	item_count: 63001	cate_count: 801	example_count: 1689188
user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
#       (user_count, item_count, cate_count, example_count))

# sort by id
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)


reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
cate_list = np.array(meta_df['categories'], dtype='int32')

# save pkl
with open('../raw_data/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
