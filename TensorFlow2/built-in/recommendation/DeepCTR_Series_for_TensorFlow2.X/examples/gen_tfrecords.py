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
#from npu_bridge.npu_init import *
import tensorflow as tf

def make_example(line, sparse_feature_name, dense_feature_name, label_name):
    features = {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[1][feat])])) for feat in
                sparse_feature_name}
    features.update(
        {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in dense_feature_name})
    features[label_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][label_name]]))
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, df, sparse_feature_names, dense_feature_names, label_name):
    writer = tf.python_io.TFRecordWriter(filename)
    for line in df.iterrows():
        ex = make_example(line, sparse_feature_names, dense_feature_names, label_name)
        writer.write(ex.SerializeToString())
    writer.close()

# write_tfrecord('./criteo_sample.tr.tfrecords',train,sparse_features,dense_features,'label')
# write_tfrecord('./criteo_sample.te.tfrecords',test,sparse_features,dense_features,'label')

