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
"""Basic Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import argparse
import random

import tensorflow as tf


def update_argparser(parser):
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=32)
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=32)
  parser.add_argument(
      '--num-data-threads',
      help='Number of threads for data transformation',
      type=int,
      default=8)
  parser.add_argument(
      '--shuffle-buffer-size',
      help='Buffer size for data shuffling',
      type=int,
      default=10000)
  parser.add_argument(
      '--prefetch-buffer-size',
      help='Buffer size for batch prefetching',
      type=int,
      default=2)


def _extract(mode, params):
  dataset = tf.data.Dataset.range(10000)
  dataset = tf.data.Dataset.zip((dataset, dataset))
  return dataset


def _transform(dataset, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(params.shuffle_buffer_size)
    dataset = dataset.repeat()

  def _preprocess(feature, label):
    feature = tf.to_float(feature)
    label = tf.to_float(label)
    if mode == tf.estimator.ModeKeys.TRAIN:
      feature += tf.random_normal([])
      label += tf.random_normal([])
    feature = tf.expand_dims(feature, -1)
    label = tf.expand_dims(label, -1)
    return {'feature': feature}, {'label': label}

  dataset = dataset.map(
      _preprocess,
      num_parallel_calls=params.num_data_threads,
  )
  batch_size = {
      tf.estimator.ModeKeys.TRAIN: params.train_batch_size,
      tf.estimator.ModeKeys.EVAL: params.eval_batch_size,
  }[mode]
  drop_remainder = {
      tf.estimator.ModeKeys.TRAIN: True,
      tf.estimator.ModeKeys.EVAL: False,
  }[mode]
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def _load(dataset, mode, params):
  dataset = dataset.prefetch(params.prefetch_buffer_size)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


def input_fn_tplt(mode,
                  params,
                  extract=_extract,
                  transform=_transform,
                  load=_load):
  dataset = extract(mode, params)
  dataset = transform(dataset, mode, params)
  dataset = load(dataset, mode, params)
  return dataset

input_fn = lambda mode, params: (
    input_fn_tplt(mode, params, _extract, _transform, _load))


def predict_input_fn():
  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')
  feature_spec = {
      'feature': tf.FixedLenFeature(shape=[1], dtype=tf.float32),
  }
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(
      features=features,
      receiver_tensors={
          tf.saved_model.signature_constants.REGRESS_INPUTS:
              serialized_tf_example
      })


def _create_tf_example():
  feature = tf.train.Feature(
      float_list=tf.train.FloatList(value=[random.randrange(10000)]))
  example = tf.train.Example(
      features=tf.train.Features(feature={'feature': feature}))
  print(example)
  return example.SerializeToString()


def test_saved_model():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model-dir', help='GCS location to load exported model', required=True)
  args = parser.parse_args()

  with tf.Session(config=npu_config_proto(), graph=tf.Graph()) as sess:
    metagraph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
    signature_def = metagraph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensor = sess.graph.get_tensor_by_name(
        signature_def.inputs['inputs'].name)
    output_tensor = sess.graph.get_tensor_by_name(
        signature_def.outputs['outputs'].name)
    print(output_tensor.eval(feed_dict={input_tensor: [_create_tf_example()]}))


if __name__ == '__main__':
  test_saved_model()
