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
tf.disable_eager_execution()
import numpy as np
from tqdm import tqdm

# Placeholder for internal import. Do not remove this line.
from datasets.google_landmarks_dataset import googlelandmarks as gld
from datasets.revisited_op import dataset
from model import delf_model


FLAGS = flags.FLAGS
flags.DEFINE_string('output_path', '/tmp/delf', 'WithTensorBoard output_path.')
flags.DEFINE_string('data_path', '/tmp/data', 'data path.')
flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
flags.DEFINE_boolean('block3_strides', True, 'Whether to use block3_strides.')
flags.DEFINE_boolean('use_augmentation', True,
                     'Whether to use ImageNet style augmentation.')
flags.DEFINE_integer('image_size', 321, 'Size of each image side to use.')
flags.DEFINE_integer('topk', 20, 'topk result.')
flags.DEFINE_boolean('use_autoencoder', True,
                     'Whether to train an autoencoder.')
flags.DEFINE_integer(
    'autoencoder_dimensions', 128,
    'Number of dimensions of the autoencoder. Used only if'
    'use_autoencoder=True.')
flags.DEFINE_integer(
    'local_feature_map_channels', 1024,
    'Number of channels at backbone layer used for local feature extraction. '
    'Default value 1024 is the number of channels of block3. Used only if'
    'use_autoencoder=True.')


def eval_step(model, image):
  images = tf.expand_dims(image, axis=0)

  # Make a forward pass to calculate prelogits.
  (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
    dim_expanded_features, dim_reduced_features) = model.global_and_local_forward_pass(images)

  return attn_scores, dim_reduced_features, desc_prelogits


def read_and_decode(example, image_size=321):
    feature_description = {
        'image/colorspace': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = gld.NormalizeImages(image, pixel_value_scale=128.0, pixel_value_offset=128.0)
    image = tf.image.resize(image, [image_size, image_size])
    image.set_shape([image_size, image_size, 3])
    return image


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('data_path= %s', FLAGS.data_path)
  logging.info('output_path= %s', FLAGS.output_path)

  image_size = FLAGS.image_size
  tf.reset_default_graph()
  # Setup session
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config_proto.allow_soft_placement = True
  ## added for enabling mix precision and loss scale
  #config_proto.graph_options.rewrite_options.auto_mixed_precision = 1
  # config_proto = tf.ConfigProto(device_count={'GPU':0})
  tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))

  # ------------------------------------------------------------
  # Create validation operation.
  model = delf_model.Delf(
      block3_strides=FLAGS.block3_strides,
      name='DELF',
      use_dim_reduction=FLAGS.use_autoencoder,
      reduced_dimension=FLAGS.autoencoder_dimensions,
      dim_expand_channels=FLAGS.local_feature_map_channels)
  # model.init_classifiers(model.num_classes)
  image_data = tf.placeholder(tf.float32, shape=(321, 321, 3))
  attn_scores, val_features, val_global_features = eval_step(model, image_data)

  # ------------------------------------------------------------
  # Create a checkpoint directory to store the checkpoints.
  saver = tf.train.Saver(max_to_keep=3)
  model_dir = os.path.join(FLAGS.data_path, "best_ckpts")

  global_step_value = 0
  global_step = tf.train.get_or_create_global_step()
  tfs.run(tf.global_variables_initializer())
  tfs.run(tf.local_variables_initializer())
  model.backbone.log_weights()

  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  saver.restore(tfs, latest_ckpt)
  logging.info("Restore from ckpt:{}".format(latest_ckpt))

  # ------------------------------------------------------------
  # Read list of query images from dataset file.
  print('Reading list of query images and boxes from dataset file...')
  dataset_dir = os.path.join(FLAGS.data_path, "test_data")
  dataset_file_path = os.path.join(dataset_dir, "gnd_roxford5k.mat")
  query_list, index_list, ground_truth = dataset.ReadDatasetFile(dataset_file_path)
  num_images = len(query_list)
  db_path = os.path.join(dataset_dir, "db.npy")
  eval_start_time = time.time()

  images_dir = os.path.join(dataset_dir, "oxford5k_tfrecords")
  print("Inference for {} image files in database.".format(len(index_list)))
  if not os.path.exists(db_path):
    filenames = tf.io.gfile.glob(os.path.join(images_dir, "db*"))
    filenames.sort()
    db_dataset = tf.data.TFRecordDataset(filenames).map(read_and_decode)
    db_inter = db_dataset.make_initializable_iterator()
    tfs.run(db_inter.initializer)
    image_iter = db_inter.get_next()
    db_feature_list = list()
    for index,index_name in tqdm(enumerate(index_list)):
      image_data_ = tfs.run(image_iter)
      attn_scores_, val_global_features_ = tfs.run([attn_scores, val_global_features], feed_dict={image_data: image_data_})
      db_feature_list.append(val_global_features_[0])
    db_np = np.array(db_feature_list)
    np.save(db_path.split(".")[0], db_np)
  else:
    db_np = np.load(db_path)

  print("Inference for {} image files for queries.".format(len(query_list)))
  avg_map = 0.0
  filenames = tf.io.gfile.glob(os.path.join(images_dir, "query*"))
  query_dataset = tf.data.TFRecordDataset(filenames).map(read_and_decode)
  query_inter = query_dataset.make_initializable_iterator()
  tfs.run(query_inter.initializer)
  query_image_iter = query_inter.get_next()
  for index, query_name in tqdm(enumerate(query_list)):
    image_data_ = tfs.run(query_image_iter)
    attn_scores_, val_global_features_ = tfs.run([attn_scores, val_global_features], feed_dict={image_data: image_data_})
    query_feature = val_global_features_[0]
    distance_list = list()
    for db_item in db_np:
        distance = np.linalg.norm(query_feature - db_item)
        distance_list.append(distance)
    sorted_indexes = np.array(distance_list).argsort().tolist()
    target_labels = np.concatenate((ground_truth[index]['easy'], ground_truth[index]['hard']))
    for hit_label in sorted_indexes[:FLAGS.topk]:
      if hit_label in target_labels:
        avg_map += 1.0
        break
  avg_map /= len(query_list)
  eval_cost = time.time() - eval_start_time
  logging.info("Eval cost in {} seconds, MAP={}".format(eval_cost, avg_map))


if __name__ == '__main__':
  app.run(main)
