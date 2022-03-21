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
import os
from pathlib import Path

import tensorflow as tf


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def get_dataset_generator(tfrecords_paths,
                          preprocess,
                          shuffle_buffer_size,
                          batch_size,
                          drop_remainder=False,
                          prefetch_buffer_size=None,
                          num_shards=None,
                          shard_index=None):
    data = tf.data.TFRecordDataset(tfrecords_paths)
    if num_shards:
        data = data.shard(num_shards, shard_index)
    if shuffle_buffer_size:
        data = data.map(preprocess).shuffle(buffer_size=shuffle_buffer_size).batch(
            batch_size, drop_remainder=drop_remainder).repeat()
    else:
        data = data.map(preprocess).batch(batch_size, drop_remainder=drop_remainder).repeat()
     
    if prefetch_buffer_size:
        data = data.prefetch(prefetch_buffer_size)
    iterator = tf.data.Iterator.from_structure(data.output_types,
                                               data.output_shapes)
    init_op = iterator.make_initializer(data)
    return init_op, iterator


def build_tfrecords(output_dir_path, output_prefix, num_shards, all_examples):
    if isinstance(output_dir_path, Path):
        output_dir_path = str(output_dir_path)
    if not tf.io.gfile.exists(output_dir_path):
        tf.io.gfile.makedirs(output_dir_path)
        print('[INFO] %s does not exist, it has been created' % output_dir_path)
    else:
        if len(tf.io.gfile.listdir(output_dir_path)) == 0:
            print('[INFO] %s already exist' % output_dir_path)
        else:
            print('[WARNING] %s is not empty!' % output_dir_path)

    tfrecord_paths = []
    for shard_id, examples in enumerate(all_examples):
        tfrecord_paths.append(
            os.path.join(
                output_dir_path, '%s_%05d-of-%05d.tfrecord' %
                (output_prefix, shard_id + 1, num_shards)))
        print('[INFO] Shard %d / %d is being generated ...... ' %
              (shard_id + 1, num_shards),
              end='',
              flush=True)
        with tf.io.TFRecordWriter(tfrecord_paths[-1]) as tf_writer:
            for example in examples:
                tf_writer.write(example.SerializeToString())
        print('Finish')

    return tfrecord_paths