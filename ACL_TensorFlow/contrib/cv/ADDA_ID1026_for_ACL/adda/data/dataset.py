"""
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
"""
import os
import adda
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import FeedingQueueRunner
AUTOTUNE = tf.data.experimental.AUTOTUNE

class DatasetGroup(object):

    def __init__(self, name, path=None, download=False):
        self.name = name
        if path is None:
            path = os.path.join(os.getcwd(), 'data')
        self.path = path
        if download:
            self.download()

    def get_path(self, *args):
        return os.path.join(self.path, self.name, *args)

    def download(self):
        """Download the dataset(s).

        This method only performs the download if necessary. If the dataset
        already resides on disk, it is a no-op.
        """
        pass


class ImageDataset(object):

    def __init__(self, images, labels, image_shape=None, label_shape=None,
                 shuffle=True):
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.shuffle = shuffle
        self.length = len(images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        inds = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(inds)
        for ind in inds:
            yield self.images[ind], self.labels[ind]

    def feed(self, im, label, epochs=None):
        epochs_elapsed = 0
        while epochs is None or epochs_elapsed < epochs:
            for entry in self:
                yield {im: entry[0], label: entry[1]}
            epochs_elapsed += 1

    def tf_ops_data(self, epoch, batch_size):
        def op():
            pass
        op.default_image_size = 28
        op.num_channels = 1
        op.mean = None
        op.bgr = False
        op.range = None
        op.mean = None
        op.bgr = False
        image_ds = tf.data.Dataset.from_tensor_slices(self.images)
        image_ds = image_ds.map(lambda x: adda.models.preprocessing(x, op))
        label_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        image_label_ds =tf.data.Dataset.zip((image_ds, label_ds))
        ds = image_label_ds.cache()
        ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.length,count=epoch,seed=0))
        ds = ds.batch(batch_size,drop_remainder=True)
        ds = ds.prefetch(buffer_size = AUTOTUNE)
        iterator = ds.make_one_shot_iterator()
        image_batchs, label_batchs = iterator.get_next()
        return image_batchs, label_batchs


    def tf_ops(self, capacity=32):
        self.im = tf.placeholder(tf.float32, shape=self.image_shape)
        self.label = tf.placeholder(tf.int32, shape=self.label_shape)
        if self.image_shape is None or self.label_shape is None:
            shapes = None
        else:
            shapes = [self.image_shape, self.label_shape]
        queue = tf.FIFOQueue(capacity, [tf.float32, tf.int32], shapes=shapes)
        enqueue_op = queue.enqueue([self.im, self.label])
        fqr = FeedingQueueRunner(queue, [enqueue_op],
                                 feed_fns=[self.feed(self.im, self.label).__next__])
        tf.train.add_queue_runner(fqr)
        return queue.dequeue()


class FilenameDataset(object):

    def tf_ops(self, capacity=32):
        im, label = tf.train.slice_input_producer(
            [tf.constant(self.images), tf.constant(self.labels)],
            capacity=capacity,
            shuffle=True)
        im = tf.read_file(im)
        im = tf.image.decode_image(im, channels=3)
        return im, label


datasets = {}


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)
