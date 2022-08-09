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
"""Extracts DELF features for query images from Revisited Oxford/Paris datasets.

Note that query images are cropped before feature extraction, as required by the
evaluation protocols of these datasets.

The program checks if descriptors already exist, and skips computation for
those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import argparse
import os
import sys
import time

from absl import app
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from protos import delf_config_pb2
from detect_to_retrieve import feature_io
from detect_to_retrieve import utils
from datasets.revisited_op import dataset
from detect_to_retrieve import extractor

cmd_args = None

# Extensions.
_DELF_EXTENSION = '.delf'
_IMAGE_EXTENSION = '.jpg'


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read list of query images from dataset file.
  print('Reading list of query images and boxes from dataset file...')
  query_list, _, ground_truth = dataset.ReadDatasetFile(
      cmd_args.dataset_file_path)
  num_images = len(query_list)
  print(f'done! Found {num_images} images')

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(cmd_args.delf_config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(cmd_args.output_features_dir):
    tf.io.gfile.makedirs(cmd_args.output_features_dir)

  # Setup session
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config_proto.allow_soft_placement = True
  # config_proto = tf.ConfigProto(device_count={'GPU':0})
  with tf.Session(config=npu_config_proto(config_proto=config_proto)) as sess:
    # Load model.
    tf.saved_model.loader.load(sess, ["serve"], config.model_path)
    graph = tf.get_default_graph()
    scale_factors = tf.placeholder(shape=[2], dtype=tf.float32, name='scale_factors')
    boxes_tensor = graph.get_tensor_by_name('boxes:0')
    features_tensor = graph.get_tensor_by_name('features:0')
    scales_tensor = graph.get_tensor_by_name('scales:0')
    scores_tensor = graph.get_tensor_by_name('scores:0')
    locations, local_descriptors, attention = extractor.extrator_fn(config, boxes_tensor, features_tensor, scores_tensor, scale_factors)

    # debug
    rf_boxes_tensor = graph.get_tensor_by_name('rf_boxes:0')
    attention_prob_tensor = graph.get_tensor_by_name('attention_prob:0')
    feature_map_tensor = graph.get_tensor_by_name('feature_map:0')
    indices_tensor = graph.get_tensor_by_name('indices:0')

    start = time.time()
    for i in range(num_images):
      print("Processing image:{}".format(i))
      query_image_name = query_list[i]
      input_image_filename = os.path.join(cmd_args.images_dir,
                                          query_image_name + _IMAGE_EXTENSION)
      output_feature_filename = os.path.join(cmd_args.output_features_dir,
                                             query_image_name + _DELF_EXTENSION)
      if tf.io.gfile.exists(output_feature_filename):
        print(f'Skipping {query_image_name}')
        continue

      # Crop query image according to bounding box.
      bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
      image = np.array(utils.RgbLoader(input_image_filename).crop(bbox))
      preprocess_result = extractor.preprocess_img(config, image)
      if preprocess_result is None:
        locations_out = np.array([])
        descriptors_out = np.array([])
        feature_scales_out = np.array([])
        attention_out = np.array([])
      else:
        # Extract and save features.
        (resized_image, scale_factors_value) = preprocess_result
        feed_data = {
                'input_image:0': resized_image,
                'input_scales:0': list(config.image_scales),
                'input_max_feature_num:0': config.delf_local_config.max_feature_num,  # 1000
                'input_abs_thres:0': config.delf_local_config.score_threshold,  # 100.0
                'scale_factors:0': scale_factors_value
            }
        locations_out, descriptors_out, feature_scales_out, attention_out= sess.run(
                [locations, local_descriptors, scales_tensor, attention], feed_dict=feed_data)
    
        # rf_boxes_value, attention_prob_value, feature_map_value, indices_value = sess.run(
        #         [rf_boxes_tensor, attention_prob_tensor, feature_map_tensor, indices_tensor], feed_dict=feed_data)
        # print(rf_boxes_value.shape, attention_prob_value.shape, feature_map_value.shape, indices_value.shape)
      feature_io.WriteToFile(output_feature_filename, locations_out,
                             feature_scales_out, descriptors_out, attention_out)

    elapsed = (time.time() - start)
    print('Processed %d query images in %f seconds' % (num_images, elapsed))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--delf_config_path',
      type=str,
      default='/tmp/delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--images_dir',
      type=str,
      default='/tmp/images',
      help="""
      Directory where dataset images are located, all in .jpg format.
      """)
  parser.add_argument(
      '--output_features_dir',
      type=str,
      default='/tmp/features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

