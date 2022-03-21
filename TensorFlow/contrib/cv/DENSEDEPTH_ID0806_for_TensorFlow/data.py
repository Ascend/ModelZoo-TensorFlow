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

import tensorflow as tf
from augment import BasicPolicy_tf
import os


def nyu_resize(img, resolution=480, padding=6):
    return tf.image.resize(img, (resolution, int(resolution * 4 / 3)))


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'data_x': tf.FixedLenFeature([], tf.string),
        'data_y': tf.FixedLenFeature([], tf.string)})

    x = tf.image.decode_jpeg(features['data_x'], channels=3)
    y = tf.image.decode_png(features['data_y'], channels=1)

    x = tf.clip_by_value(x / 255, 0, 1)
    y = tf.clip_by_value(y / 255 * 1000.0, 10, 1000.0)

    y = 1000.0 / y

    x = nyu_resize(x, 480)
    y = nyu_resize(y, 240)

    return x, y


def _parse_function(x, y, is_addnoise=False):
    policy = BasicPolicy_tf(color_change_ratio=0.50, mirror_ratio=0.50,
                            add_noise_peak=0 if not is_addnoise else 20)

    x, y = policy(x, y)

    return x, y


def create_dataset(train_tfrecords, minibatch_size, is_distributed):
    print('Setting up dataset source from', train_tfrecords)
    buffer_mb = 256
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb << 20)
    if is_distributed:
        rank_size = int(os.getenv('RANK_SIZE'))
        rank_id = int(os.getenv('RANK_ID'))
        print('RANK_SIZE=', rank_size, ' RANK_ID=', rank_id)
        dset = dset.shard(rank_size, rank_id)
    buf_size = 50688
    dset = dset.shuffle(buffer_size=buf_size)
    dset = dset.repeat()
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
    dset = dset.map(lambda x, y: _parse_function(x, y))
    dset = dset.batch(minibatch_size, drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    # it = dset.make_one_shot_iterator()
    return dset
