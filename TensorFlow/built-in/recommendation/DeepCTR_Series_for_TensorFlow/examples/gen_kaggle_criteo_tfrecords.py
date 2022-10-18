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

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

TRAIN_LINE_COUINT = 45840617
TEST_LINE_COUNT = 6042135
CHUNKSIZE=1000000

label = 'label'
dense_columns = [f'I{i}' for i in range(1, 14)]
categorical_columns = [f'C{i}' for i in range(1, 27)]
columns = [label] + dense_columns + categorical_columns

def make_example(line, sparse_feature_name, dense_feature_name, label_name):
    features = {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[1][feat])])) for feat in
                sparse_feature_name}
    features.update(
        {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in dense_feature_name})
    features[label_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][label_name]]))
    return tf.train.Example(features=tf.train.Features(feature=features))

src_filename = "/data/criteo/train/train.txt"
base_dir = os.path.dirname(src_filename)
csv_reader = pd.read_csv(src_filename, sep='\t', header=None, names=columns, chunksize=CHUNKSIZE)
chunks = []

for idx, data in enumerate(csv_reader):
    print(idx, ' ', len(data))
    data[dense_columns] = data[dense_columns].fillna(0)
    data[categorical_columns] = data[categorical_columns].fillna('-1')

    # default type is int64 float64 object
    data[label] = data[label].astype(np.int32)
    data[dense_columns] = data[dense_columns].astype(np.int32)
    chunks.append(data)

df = pd.concat(chunks, axis=0, ignore_index=True)
print("total length:", len(df))

for feat in categorical_columns:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_columns] = mms.fit_transform(df[dense_columns])

df[label] = df[label].astype(np.int32)
df[dense_columns] = df[dense_columns].astype(np.float32)
df[categorical_columns] = df[categorical_columns].astype(np.int32)

print("start to split train and test...")
train, test = train_test_split(df, test_size=0.1, random_state=2022)

train_loop_size = len(train) // CHUNKSIZE
train_remain_size = len(train) % CHUNKSIZE
test_loop_size = len(test) // CHUNKSIZE
test_remain_size = len(test) % CHUNKSIZE

print("The length of train:", train_loop_size)
for idx in range(train_loop_size):
    data = train.iloc[idx*CHUNKSIZE:(idx+1)*CHUNKSIZE, :]
    out_filename = os.path.join(base_dir, "train_part_{}.csv".format(idx))
    data.to_csv(out_filename, sep='\t', index=False)
    tf_filename = os.path.join(base_dir, "train_part_{}.tfrecord".format(idx))
    writer = tf.io.TFRecordWriter(tf_filename)
    for line in df.iterrows():
        ex = make_example(line, categorical_columns, dense_columns, label)
        writer.write(ex.SerializeToString())
    writer.close()

if train_remain_size > 0:
    data = train.iloc[-train_remain_size:, :]
    out_filename = os.path.join(base_dir, "train_part_{}.csv".format(idx))
    data.to_csv(out_filename, sep='\t', index=False)
    tf_filename = os.path.join(base_dir, "train_part_{}.tfrecord".format(idx))
    writer = tf.io.TFRecordWriter(tf_filename)
    for line in df.iterrows():
        ex = make_example(line, categorical_columns, dense_columns, label)
        writer.write(ex.SerializeToString())
    writer.close()

print("The length of test:", test_loop_size)
for idx in range(test_loop_size):
    data = train.iloc[idx*CHUNKSIZE:(idx+1)*CHUNKSIZE, :]
    out_filename = os.path.join(base_dir, "test_part_{}.csv".format(idx))
    data.to_csv(out_filename, sep='\t', index=False)
    tf_filename = os.path.join(base_dir, "test_part_{}.tfrecord".format(idx))
    writer = tf.io.TFRecordWriter(tf_filename)
    for line in df.iterrows():
        ex = make_example(line, categorical_columns, dense_columns, label)
        writer.write(ex.SerializeToString())
    writer.close()

if test_remain_size > 0:
    data = train.iloc[-test_remain_size:, :]
    out_filename = os.path.join(base_dir, "test_part_{}.csv".format(idx))
    data.to_csv(out_filename, sep='\t', index=False)
    tf_filename = os.path.join(base_dir, "test_part_{}.tfrecord".format(idx))
    writer = tf.io.TFRecordWriter(tf_filename)
    for line in df.iterrows():
        ex = make_example(line, categorical_columns, dense_columns, label)
        writer.write(ex.SerializeToString())
    writer.close()

