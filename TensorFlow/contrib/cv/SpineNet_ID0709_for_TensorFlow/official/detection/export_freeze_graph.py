# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=line-too-long
r"""A binary to export the inference graph of TPU detection model.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('/mnt/hgfs/share/tpu_small/models')
sys.path.append('/mnt/hgfs/share/tpu_small/models/official/efficientnet')

from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph

from configs import factory
from serving import detection
from serving import inputs
from hyperparameters import params_dict
from tensorflow.python.framework import graph_util  # pylint: disable=g-direct-tensorflow-import


FLAGS = flags.FLAGS

flags.DEFINE_string('export_dir', None, 'The export directory.')
flags.DEFINE_string(
    'model', 'retinanet',
    'Model to run: `retinanet`, `mask_rcnn` or `shapemask`.')
flags.DEFINE_string(
    'config_file', '',
    'The JSON/YAML parameter file which serves as the config template.')
flags.DEFINE_string(
    'params_override', '',
    'The JSON/YAML file or string which specifies the parameter to be overriden'
    ' on top of `config_file` template.')
flags.DEFINE_integer('batch_size', 1, 'The batch size.')
flags.DEFINE_string(
    'input_type', 'image_tensor',
    'One of `raw_image_tensor`, `image_tensor`, `image_bytes`, `tf_example`.')
flags.DEFINE_string('input_name', 'input', 'The name of the input node.')
flags.DEFINE_string(
    'input_image_size', '640,640',
    'The comma-separated string of two integers representing the height,width '
    'of the input to the model.')
flags.DEFINE_boolean(
    'optimize_graph', False, 'Whether or not optimize the graph.')
flags.DEFINE_boolean(
    'output_image_info', True, 'Whether or not output image_info node.')
flags.DEFINE_boolean(
    'output_normalized_coordinates', False,
    'Whether or not output boxes in normalized coordinates.')
flags.DEFINE_boolean(
    'cast_num_detections_to_float', False,
    'Whether or not cast the number of detections to float type.')


def main(argv):
  del argv  # Unused.

  params = factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  # Use `is_strict=False` to load params_override with run_time variables like
  # `train.num_shards`.
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=False)
  params.validate()
  params.lock()

  image_size = [int(x) for x in FLAGS.input_image_size.split(',')]

  g = tf.Graph()
  with g.as_default():
    # Build the input.
    _, features = inputs.build_serving_input(
        input_type=FLAGS.input_type,
        batch_size=FLAGS.batch_size,
        desired_image_size=image_size,
        stride=(2 ** params.architecture.max_level))

    # Build the model.
    print(' - Building the graph...')
    if FLAGS.model in ['retinanet', 'mask_rcnn', 'shapemask']:
      graph_fn = detection.serving_model_graph_builder(
          FLAGS.output_image_info,
          FLAGS.output_normalized_coordinates,
          FLAGS.cast_num_detections_to_float)
    else:
      raise ValueError(
          'The model type `{}` is not supported.'.format(FLAGS.model))

    predictions = graph_fn(features, params)

    # Add a saver for checkpoint loading.
    tf.train.Saver()

    inference_graph_def = g.as_graph_def()
    optimized_graph_def = inference_graph_def

    if FLAGS.optimize_graph:
      print(' - Optimizing the graph...')
      # Trim the unused nodes in the graph.
      output_nodes = [output_node.op.name
                      for output_node in predictions.values()]
      # TODO(pengchong): Consider to use `strip_unused_lib.strip_unused` and/or
      # `optimize_for_inference_lib.optimize_for_inference` to trim the graph.
      # Use `optimize_for_inference` if we decide to export the frozen graph
      # (graph + checkpoint) and want explictily fold in batchnorm variables.
      optimized_graph_def = graph_util.remove_training_nodes(
          optimized_graph_def, output_nodes)

  print('features:', features)
  print('predictions:', predictions)
  print(' - Saving the graph...')
  tf.train.write_graph(
      optimized_graph_def, FLAGS.export_dir, 'inference_graph.pbtxt')
  # tf.train.write_graph(
  #     optimized_graph_def, FLAGS.export_dir, 'inference_graph.pb')
  print(' - Freeze the graph...')
  freeze_graph.freeze_graph(
      input_graph=os.path.join(FLAGS.export_dir, 'inference_graph.pbtxt'),
      input_saver='',
      input_binary=False,
      input_checkpoint=os.path.join(FLAGS.export_dir, 'model.ckpt-231500'),
      output_node_names='retinanet/class_net/class-predict/BiasAdd, DetectionClasses, NumDetections, DetectionBoxes, DetectionScores, ImageInfo',  # graph outputs node
      restore_op_name='save/restore_all',
      filename_tensor_name='save/Const:0',
      output_graph=os.path.join(FLAGS.export_dir, 'inference_graph_freeze.pb'),  # graph outputs name
      clear_devices=False,
      initializer_nodes="")
  print(' - Done!')


if __name__ == '__main__':
  flags.mark_flag_as_required('export_dir')
  flags.mark_flag_as_required('model')
  tf.app.run(main)
