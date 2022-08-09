# Lint as: python3

# coding=utf-8
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
# ==============================================================================
"""Training script for DELF/G on Google Landmarks Dataset.

Uses classification loss, with MirroredStrategy, to support running on multiple
GPUs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os
import time
import itertools
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
### tf.enable_eager_execution()
import numpy as np
from tqdm import tqdm

# Placeholder for internal import. Do not remove this line.
from datasets.google_landmarks_dataset import googlelandmarks as gld
from datasets.revisited_op import dataset
from model import delf_model


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_file_path', None, 'file_path oxford5k images.')
flags.DEFINE_string('images_dir', None, 'image directory.')
flags.DEFINE_string('tfrecords_dir', None, 'image directory.')
flags.DEFINE_integer('num_shards', 128, 'Number of shards in output data.')


def eval_step(model, image_data, image_size=321):
  # Decode and resize RGB JPEG.
  image = tf.io.decode_jpeg(image_data, channels=3)
  image = gld.NormalizeImages(image, pixel_value_scale=128.0, pixel_value_offset=128.0)
  image = tf.image.resize(image, [image_size, image_size])
  image.set_shape([image_size, image_size, 3])
  images = tf.expand_dims(image, axis=0)

  return attn_scores, dim_reduced_features, desc_prelogits


def _process_image(filename):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.jpg'.

  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  Raises:
    ValueError: if parsed image has wrong number of dimensions or channels.
  """
  # Read the image file.
  with tf.io.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = tf.io.decode_jpeg(image_data, channels=3)

  # Check that image converted to RGB
  if len(image.shape) != 3:
    raise ValueError('The parsed image number of dimensions is not 3 but %d' %
                     (image.shape))
  height = image.shape[0]
  width = image.shape[1]
  if image.shape[2] != 3:
    raise ValueError('The parsed image channels is not 3 but %d' %
                     (image.shape[2]))
  return image_data, height, width


def _convert_to_example(file_id, image_buffer, height, width, label=None):
  """Build an Example proto for the given inputs.

  Args:
    file_id: string, unique id of an image file, e.g., '97c0a12e07ae8dd5'.
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    label: integer, the landmark id and prediction label.

  Returns:
    Example proto.
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  features = {
      'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/id': _bytes_feature(file_id.encode('utf-8')),
      'image/encoded': _bytes_feature(image_buffer)
  }
  if label is not None:
    features['image/class/label'] = _int64_feature(label)
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
  # ------------------------------------------------------------
  # Read list of query images from dataset file.
  print('Reading list of query images and boxes from dataset file...')
  query_list, index_list, ground_truth = dataset.ReadDatasetFile(FLAGS.dataset_file_path)
  if not os.path.exists(FLAGS.tfrecords_dir):
    os.mkdir(FLAGS.tfrecords_dir)
  query_file = os.path.join(FLAGS.tfrecords_dir, 'query-%.5d-of-%.5d' % (0, 1))
  writer = tf.io.TFRecordWriter(query_file)
  print('Generating queries and writing file ', query_file)
  for image_name in query_list:
    image_path = os.path.join(FLAGS.images_dir, "{}.jpg".format(image_name))
    image_buffer, height, width = _process_image(image_path)
    example = _convert_to_example(image_name, image_buffer, height, width)
    writer.write(example.SerializeToString())
  writer.close()

  image_paths = [os.path.join(FLAGS.images_dir, "{}.jpg".format(item)) for item in index_list]
  spacing = np.linspace(0, len(image_paths), FLAGS.num_shards + 1, dtype=np.int)
  for shard in range(FLAGS.num_shards):
    output_file = os.path.join(
        FLAGS.tfrecords_dir,
        'db-%.5d-of-%.5d' % (shard, FLAGS.num_shards))
    writer = tf.io.TFRecordWriter(output_file)
    print('Processing shard ', shard, ' and writing file ', output_file)
    for i in range(spacing[shard], spacing[shard + 1]):
      image_buffer, height, width = _process_image(image_paths[i])
      image_name = os.path.basename(image_paths[i]).split(".")[0]
      example = _convert_to_example(image_name, image_buffer, height, width)
      writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  app.run(main)

