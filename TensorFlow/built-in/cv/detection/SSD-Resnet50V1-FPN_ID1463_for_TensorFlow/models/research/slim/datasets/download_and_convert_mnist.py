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

'Downloads and converts MNIST data to TFRecords of TF-Example protos.\n\nThis module downloads the MNIST data, uncompresses it, reads the files\nthat make up the MNIST data and creates two TFRecord datasets: one for train\nand one for test. Each TFRecord dataset is comprised of a set of TF-Example\nprotocol buffers, each of which contain a single image and label.\n\nThe script should take about a minute to run.\n\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.core.protobuf import config_pb2
import gzip
import os
import sys
import numpy as np
from six.moves import urllib
import tensorflow as tf
from datasets import dataset_utils

def npu_session_config_init(session_config=None):
    if ((not isinstance(session_config, config_pb2.ConfigProto)) and (not issubclass(type(session_config), config_pb2.ConfigProto))):
        session_config = config_pb2.ConfigProto()
    if (isinstance(session_config, config_pb2.ConfigProto) or issubclass(type(session_config), config_pb2.ConfigProto)):
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return session_config
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'
_IMAGE_SIZE = 28
_NUM_CHANNELS = 1
_CLASS_NAMES = ['zero', 'one', 'two', 'three', 'four', 'five', 'size', 'seven', 'eight', 'nine']

def _extract_images(filename, num_images):
    'Extract the images into a numpy array.\n\n  Args:\n    filename: The path to an MNIST images file.\n    num_images: The number of images in the file.\n\n  Returns:\n    A numpy array of shape [number_of_images, height, width, channels].\n  '
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read((((_IMAGE_SIZE * _IMAGE_SIZE) * num_images) * _NUM_CHANNELS))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    return data

def _extract_labels(filename, num_labels):
    'Extract the labels into a vector of int64 label IDs.\n\n  Args:\n    filename: The path to an MNIST labels file.\n    num_labels: The number of labels in the file.\n\n  Returns:\n    A numpy array of shape [number_of_labels]\n  '
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read((1 * num_labels))
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def _add_to_tfrecord(data_filename, labels_filename, num_images, tfrecord_writer):
    'Loads data from the binary MNIST files and writes files to a TFRecord.\n\n  Args:\n    data_filename: The filename of the MNIST images.\n    labels_filename: The filename of the MNIST labels.\n    num_images: The number of images in the dataset.\n    tfrecord_writer: The TFRecord writer to use for writing.\n  '
    images = _extract_images(data_filename, num_images)
    labels = _extract_labels(labels_filename, num_images)
    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)
        with tf.Session('', config=npu_session_config_init()) as sess:
            for j in range(num_images):
                sys.stdout.write(('\r>> Converting image %d/%d' % ((j + 1), num_images)))
                sys.stdout.flush()
                png_string = sess.run(encoded_png, feed_dict={image: images[j]})
                example = dataset_utils.image_to_tfexample(png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
                tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(dataset_dir, split_name):
    'Creates the output filename.\n\n  Args:\n    dataset_dir: The directory where the temporary files are stored.\n    split_name: The name of the train/test split.\n\n  Returns:\n    An absolute file path.\n  '
    return ('%s/mnist_%s.tfrecord' % (dataset_dir, split_name))

def _download_dataset(dataset_dir):
    'Downloads MNIST locally.\n\n  Args:\n    dataset_dir: The directory where the temporary files are stored.\n  '
    for filename in [_TRAIN_DATA_FILENAME, _TRAIN_LABELS_FILENAME, _TEST_DATA_FILENAME, _TEST_LABELS_FILENAME]:
        filepath = os.path.join(dataset_dir, filename)
        if (not os.path.exists(filepath)):
            print(('Downloading file %s...' % filename))

            def _progress(count, block_size, total_size):
                sys.stdout.write(('\r>> Downloading %.1f%%' % ((float((count * block_size)) / float(total_size)) * 100.0)))
                sys.stdout.flush()
            (filepath, _) = urllib.request.urlretrieve((_DATA_URL + filename), filepath, _progress)
            print()
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print('Successfully downloaded', filename, size, 'bytes.')

def _clean_up_temporary_files(dataset_dir):
    'Removes temporary files used to create the dataset.\n\n  Args:\n    dataset_dir: The directory where the temporary files are stored.\n  '
    for filename in [_TRAIN_DATA_FILENAME, _TRAIN_LABELS_FILENAME, _TEST_DATA_FILENAME, _TEST_LABELS_FILENAME]:
        filepath = os.path.join(dataset_dir, filename)
        tf.gfile.Remove(filepath)

def run(dataset_dir):
    'Runs the download and conversion operation.\n\n  Args:\n    dataset_dir: The dataset directory where the dataset is stored.\n  '
    if (not tf.gfile.Exists(dataset_dir)):
        tf.gfile.MakeDirs(dataset_dir)
    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')
    if (tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename)):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    _download_dataset(dataset_dir)
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        data_filename = os.path.join(dataset_dir, _TRAIN_DATA_FILENAME)
        labels_filename = os.path.join(dataset_dir, _TRAIN_LABELS_FILENAME)
        _add_to_tfrecord(data_filename, labels_filename, 60000, tfrecord_writer)
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        data_filename = os.path.join(dataset_dir, _TEST_DATA_FILENAME)
        labels_filename = os.path.join(dataset_dir, _TEST_LABELS_FILENAME)
        _add_to_tfrecord(data_filename, labels_filename, 10000, tfrecord_writer)
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the MNIST dataset!')
