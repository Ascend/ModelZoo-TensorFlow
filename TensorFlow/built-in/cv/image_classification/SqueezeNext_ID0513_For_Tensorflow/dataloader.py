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


from __future__ import absolute_import
from npu_bridge.npu_init import *
import tensorflow as tf
import multiprocessing

def caffe_center_crop(image_encoded, image_size, training, resize_size=256):
    '\n    Emulates the center crop function used in caffe\n    :param image_encoded:\n        Jpeg string\n    :param image_size:\n        Output width and height\n    :param training:\n        Whether or not the model is training\n    :param resize_size:\n        Size to which to resize the decoded image before center croping. Default size is 256\n        to match the size used in this script:\n        https://github.com/BVLC/caffe/blob/master/examples/imagenet/create_imagenet.sh\n    :return:\n        Image of size [image_size,image_size,3]\n    '
    image = tf.image.decode_jpeg(image_encoded, channels=3)
    image = tf.image.resize_images(image, [resize_size, resize_size])
    image = tf.reshape(image, [resize_size, resize_size, 3])
    if training:
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        crop_min = tf.cast(tf.abs(((resize_size / 2) - (image_size / 2))), tf.int32)
        crop_max = (crop_min + image_size)
        image = image[crop_min:crop_max, crop_min:crop_max, :]
    return image

def _parse_function(example_proto, image_size, num_classes, training, mean_value=(123, 117, 104), method='crop'):
    '\n    Parses tf-records created with build_imagenet_data.py\n    :param example_proto:\n        Single example from tf record\n    :param image_size:\n        Output image size\n    :param num_classes:\n        Number of classes in dataset\n    :param training:\n        Whether or not the model is training\n    :param mean_value:\n        Imagenet mean to subtract from the output iamge\n    :param method:\n        How to generate the input image\n    :return:\n        Features dict containing image, and labels dict containing class index and one hot vector\n    '
    schema = {'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''), 'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=(- 1))}
    image_size = tf.cast(image_size, tf.int32)
    mean_value = tf.cast(tf.stack(mean_value), tf.float32)
    parsed_features = tf.parse_single_example(example_proto, schema)
    jpeg_image = parsed_features['image/encoded']
    if (method == 'crop'):
        image = caffe_center_crop(jpeg_image, image_size, training)
    elif (method == 'resize'):
        image = tf.image.decode_jpeg(jpeg_image)
        image = tf.image.resize_images(image, [image_size, image_size])
    else:
        raise 'unknown image process method'
    image = (image - mean_value)
    label_idx = (tf.cast(parsed_features['image/class/label'], dtype=tf.int32) - 1)
    label_vec = tf.one_hot(label_idx, num_classes)
    return ({'image': tf.reshape(image, [image_size, image_size, 3])}, {'class_idx': label_idx, 'class_vec': label_vec})

class ReadTFRecords(object):

    def __init__(self, image_size, batch_size, num_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __call__(self, glob_pattern, training=True):
        '\n        Read tf records matching a glob pattern\n        :param glob_pattern:\n            glob pattern eg. "/usr/local/share/Datasets/Imagenet/train-*.tfrecords"\n        :param training:\n            Whether or not to shuffle the data for training and evaluation\n        :return:\n            Iterator generating one example of batch size for each training step\n        '
        threads = multiprocessing.cpu_count()
        with tf.name_scope('tf_record_reader'):
            files = tf.data.Dataset.list_files(glob_pattern, shuffle=training)

            def map_func(filename):
                return tf.data.TFRecordDataset(filename)
            dataset = files.apply(tf.contrib.data.parallel_interleave(map_func, cycle_length=threads))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat((32 * self.batch_size)))
            dataset = dataset.map(map_func=(lambda example: _parse_function(example, self.image_size, self.num_classes, training=training)), num_parallel_calls=threads)
            dataset = dataset.batch(drop_remainder=True, batch_size=self.batch_size)
            dataset = dataset.prefetch(buffer_size=32)
            return dataset.make_one_shot_iterator().get_next()
