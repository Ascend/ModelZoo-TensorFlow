# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3


import tensorflow  as tf
from preprocess import preprocess_for_train
from preprocess import preprocess_for_eval

def train_parse_read(tfrecord_file):
    features = {
        "fine_label":
            tf.io.FixedLenFeature((), tf.int64),
        "coarse_label":
            tf.io.FixedLenFeature((), tf.int64),
        "image":
            tf.io.FixedLenFeature((), tf.string, default_value="")
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float64)
    label = parsed["fine_label"]
    image = preprocess_for_train(image,32,32)
    return image, label

def test_parse_read(tfrecord_file):
    features = {
        "fine_label":
            tf.io.FixedLenFeature((), tf.int64),
        "coarse_label":
            tf.io.FixedLenFeature((), tf.int64),
        "image":
            tf.io.FixedLenFeature((), tf.string, default_value="")
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float64)
    label = parsed["fine_label"]
    image = preprocess_for_eval(image,32,32)
    return image, label



def get_train_data(tf_data_path,batch_size,epoch):
    dataset = tf.data.TFRecordDataset(tf_data_path)
    dataset = dataset.map(train_parse_read, num_parallel_calls=2)
    dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    return images_batch, labels_batch


def get_test_data(tf_data_path,batch_size):
    dataset = tf.data.TFRecordDataset(tf_data_path)
    dataset = dataset.map(test_parse_read, num_parallel_calls=2)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    return images_batch, labels_batch
