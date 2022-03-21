# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# ============================================================================
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
import os

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    return tf.transpose(x, perm=[1, 2, 0])


# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return tf.transpose(x, perm=[2, 0, 1])


def resize_small_image(x):
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        true_fn=lambda: hwc_to_chw(
            tf.image.resize_images(chw_to_hwc(x), size=[256, 256], method=tf.image.ResizeMethod.BICUBIC)),
        false_fn=lambda: tf.cast(x, tf.float32)
    )


def random_crop_noised_clean(x, add_noise):
    cropped = tf.random_crop(resize_small_image(x), size=[3, 256, 256]) / 255.0 - 0.5
    return (add_noise(cropped), add_noise(cropped), cropped)


def create_dataset(train_tfrecords, minibatch_size, add_noise, is_distributed):
    print('Setting up dataset source from', train_tfrecords)
    buffer_mb = 256
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb << 20)
    if is_distributed:
        rank_size = int(os.getenv('RANK_SIZE'))
        rank_id = int(os.getenv('RANK_ID'))
        print('RANK_SIZE=', rank_size, ' RANK_ID=', rank_id)
        dset = dset.shard(rank_size, rank_id)
    dset = dset.repeat()
    buf_size = 1000
    dset = dset.prefetch(buf_size)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dset = dset.shuffle(buffer_size=buf_size)
    dset = dset.map(lambda x: random_crop_noised_clean(x, add_noise))
    dset = dset.batch(minibatch_size, drop_remainder=True)
    it = dset.make_one_shot_iterator()
    return it
