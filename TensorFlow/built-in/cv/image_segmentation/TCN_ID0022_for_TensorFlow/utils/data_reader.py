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
from npu_bridge.npu_init import *
import h5py
import random
import tensorflow as tf
import numpy as np
from .img_utils import get_images


"""
This module provides three data reader: directly from file, from h5 database, use channel

h5 database is recommended since it could enable very data feeding speed
"""


class FileDataReader(object):

    def __init__(self, data_dir, input_height, input_width, height, width,
                 batch_size):
        self.data_dir = data_dir
        self.input_height, self.input_width = input_height, input_width
        self.height, self.width = height, width
        self.batch_size = batch_size
        self.image_files = glob.glob(data_dir+'*')

    def next_batch(self, batch_size):
        sample_files = np.random.choice(self.image_files, batch_size)
        images = get_images(
            sample_files, self.input_height, self.input_width,
            self.height, self.width)
        return images


class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'], data_file['Y']
        self.gen_indexes()

    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0

    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        return self.images[cur_indexes], self.labels[cur_indexes]


class H53DDataLoader(object):

    def __init__(self, data_path, shape, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['data'], data_file['label']
        self.t_h, self.t_w, self.t_d = data_file['data'].shape
        self.d, self.h, self.w = shape[1:-1]

    def next_batch(self, batch_size):
        batches_ids = set()
        while len(batches_ids) < batch_size:
            h = random.randint(0, self.t_h-self.h)
            w = random.randint(0, self.t_w-self.w)
            d = random.randint(0, self.t_d-self.d)
            batches_ids.add((h, w, d))
        image_batches = []
        label_batches = []
        for h, w, d in batches_ids:
            image_batches.append(
                self.images[h:h+self.h, w:w+self.w, d:d+self.d])
            label_batches.append(
                self.labels[h:h+self.h, w:w+self.w, d:d+self.d])
        images = np.expand_dims(np.stack(image_batches, axis=0), axis=-1)
        images = np.transpose(images, (0, 3, 1, 2, 4))
        labels = np.stack(label_batches, axis=0)
        labels = np.transpose(labels, (0, 3, 1, 2))
        return images, labels


class QueueDataReader(object):

    def __init__(self, sess, data_dir, data_list, input_size, class_num,
                 name, data_format):
        self.sess = sess
        self.scope = name + '/data_reader'
        self.class_num = class_num
        self.channel_axis = 3
        images, labels = self.read_data(data_dir, data_list)
        images = tf.convert_to_tensor(images, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.string)
        queue = tf.train.slice_input_producer(
            [images, labels], shuffle=True, name=self.scope+'/slice')
        self.image, self.label = self.read_dataset(
            queue, input_size, data_format)

    def next_batch(self, batch_size):
        image_batch, label_batch = tf.train.shuffle_batch(
            [self.image, self.label], batch_size=batch_size,
            num_threads=4, capacity=50000, min_after_dequeue=10000,
            name=self.scope+'/batch')
        return image_batch, label_batch

    def read_dataset(self, queue, input_size, data_format):
        image = tf.image.decode_jpeg(
            tf.read_file(queue[0]), channels=3, name=self.scope+'/image')
        label = tf.image.decode_png(
            tf.read_file(queue[1]), channels=1, name=self.scope+'/label')
        image = tf.image.resize_images(image, input_size)
        label = tf.image.resize_images(label, input_size, 1)
        if data_format == 'NCHW':
            self.channel_axis = 1
            image = tf.transpose(image, [2, 0, 1])
            label = tf.transpose(label, [2, 0, 1])
        image -= tf.reduce_mean(tf.cast(image, dtype=tf.float32),
                                (0, 1), name=self.scope+'/mean')
        return image, label

    def read_data(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            images, labels = [], []
            for line in f:
                image, label = line.strip('\n').split(' ')
                images.append(data_dir + image)
                labels.append(data_dir + label)
        return images, labels

    def start(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            coord=self.coord, sess=self.sess)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

