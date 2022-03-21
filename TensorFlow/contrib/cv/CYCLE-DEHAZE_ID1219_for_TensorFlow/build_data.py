# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
"""
Make tfrecords file according to the images(.png)
"""
from npu_bridge.npu_init import *
import tensorflow as tf
import random
import os
from PIL import Image

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('X_input_dir', 'data/trainData/hazyImage',
                       'X input directory, default: data/trainData/hazyImage')  #������ͼƬ
tf.flags.DEFINE_string('Y_input_dir', 'data/trainData/clearImage',
                       'Y input directory, default: data/trainData/clearImage')  #û������ͼƬ
tf.flags.DEFINE_string('X_output_file', 'data/tfrecords/hazyImage.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/hazyImage.tfrecords')
tf.flags.DEFINE_string('Y_output_file', 'data/tfrecords/clearImage.tfrecords',
                       'Y output tfrecords file, default: data/tfrecords/clearImage.tfrecords')


def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []


  for img_file in scandir(input_dir):
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example


def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except (os.error):
    pass

  images_num = len(file_paths)

  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()


def main(unused_argv):
  '''
  main(), make tfrecords file.
  '''
  print("Convert X data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
  print("Convert Y data to tfrecords...")
  data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)


if __name__ == '__main__':
  tf.app.run()

