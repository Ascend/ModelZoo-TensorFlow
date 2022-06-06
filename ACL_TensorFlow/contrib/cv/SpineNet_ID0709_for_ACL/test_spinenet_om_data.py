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
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io
import os

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output', '', 'The path to the om output file.')
flags.DEFINE_boolean(
    'use_normalized_coordinates', False,
    'Wether or not the SavedModel output the normalized coordinates.')
flags.DEFINE_string(
    'label_map_file', '',
    'The label map file. See --label_map_format for the definition.')
flags.DEFINE_string(
    'label_map_format', 'csv',
    'The format of the label map file. Currently only support `csv` where the '
    'format of each row is: `id:name`.')
flags.DEFINE_string(
    'image_file_pattern', '',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'output_html', '/tmp/test.html',
    'The output HTML file that includes images with rendered detections.')
flags.DEFINE_integer(
    'max_boxes_to_draw', 10, 'The maximum number of boxes to draw.')
flags.DEFINE_float(
    'min_score_threshold', 0.05,
    'The minimum score thresholds in order to draw boxes.')


def main(unused_argv):
  del unused_argv
  # Load the label map.
  print(' - Loading the label map...')
  label_map_dict = {}
  if FLAGS.label_map_format == 'csv':
    with tf.gfile.Open(FLAGS.label_map_file, 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=':')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id:name` format.')
        id_index = int(row[0])
        name = row[1]
        label_map_dict[id_index] = {
            'id': id_index,
            'name': name,
        }
  else:
    raise ValueError(
        'Unsupported label map format: {}.'.format(FLAGS.label_map_format))


  image_with_detections_list = []
  image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
  for i, image_file in enumerate(image_files):
    print(' - processing image %d...' % i)

    image = Image.open(image_file)
    image = image.convert('RGB')  # needed for images with 4 channels.
    width, height = image.size

    np_image = (np.array(image.getdata())
                .reshape(height, width, 3).astype(np.uint8))
				
    num_detections = np.loadtxt(os.path.join(FLAGS.output, "{0:05d}_output_0.txt".format(i)))
    num_detections = int(num_detections)
    np_boxes = np.loadtxt(os.path.join(FLAGS.output, "{0:05d}_output_1.txt".format(i)))
    if not FLAGS.use_normalized_coordinates:
      np_image_info = np.loadtxt(os.path.join(FLAGS.output, "{0:05d}_output_4.txt".format(i)))
      np_boxes = np_boxes / np.tile(np_image_info[1:2, :], (1, 2))
    ymin, xmin, ymax, xmax = np.split(np_boxes, 4, axis=-1)
    ymin = ymin * height
    ymax = ymax * height
    xmin = xmin * width
    xmax = xmax * width
    np_boxes = np.concatenate([ymin, xmin, ymax, xmax], axis=-1)
    np_classes = np.loadtxt(os.path.join(FLAGS.output, "{0:05d}_output_2.txt".format(i)))
    np_classes = np_classes.astype(np.int32)
    np_scores = np.loadtxt(os.path.join(FLAGS.output, "{0:05d}_output_3.txt".format(i)))
    np_masks = None

    image_with_detections = (
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            np_image,
            np_boxes,
            np_classes,
            np_scores,
            label_map_dict,
            instance_masks=np_masks,
            use_normalized_coordinates=False,
            max_boxes_to_draw=FLAGS.max_boxes_to_draw,
            min_score_thresh=FLAGS.min_score_threshold))
    image_with_detections_list.append(image_with_detections)

  print(' - Saving the outputs...')
  formatted_image_with_detections_list = [
      Image.fromarray(image.astype(np.uint8))
      for image in image_with_detections_list]
  html_str = '<html>'
  image_strs = []
  for formatted_image in formatted_image_with_detections_list:
    with io.BytesIO() as stream:
      formatted_image.save(stream, format='JPEG')
      data_uri = base64.b64encode(stream.getvalue()).decode('utf-8')
    image_strs.append(
        '<img src="data:image/jpeg;base64,{}", height=800>'
        .format(data_uri))
  images_str = ' '.join(image_strs)
  html_str += images_str
  html_str += '</html>'
  with tf.gfile.GFile(FLAGS.output_html, 'w') as f:
    f.write(html_str)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('image_file_pattern')
  flags.mark_flag_as_required('output_html')
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
