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
"""Module to construct DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import numpy as np
import tensorflow as tf

from detect_to_retrieve import datum_io
from detect_to_retrieve import feature_extractor
from detect_to_retrieve import utils

# Minimum dimensions below which features are not extracted (empty
# features are returned). This applies after any resizing is performed.
_MIN_HEIGHT = 10
_MIN_WIDTH = 10


def MakeExtractor(config, sess):
  """Creates a function to extract global and/or local features from an image.

  Args:
    config: DelfConfig proto containing the model configuration.

  Returns:
    Function that receives an image and returns features.

  Raises:
    ValueError: if config is invalid.
  """
  # Assert the configuration.
  if not config.use_local_features and not config.use_global_features:
    raise ValueError('Invalid config: at least one of '
                     '{use_local_features, use_global_features} must be True')

  # Load model.
  model = tf.saved_model.loader.load(sess, ["serve"], config.model_path)
  sig_def = meta_graph.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  input_image_tensor = sig_def.inputs['input_image']
  input_scales_tensor = sig_def.inputs['input_scales']
  input_max_feature_num_tensor = sig_def.inputs['input_max_feature_num']
  iput_abs_thres_tensor = sig_def.inputs['iput_abs_thres']
  boxes_tensor = sig_def.outputs['boxes']
  features_tensor = sig_def.outputs['features']
  scales_tensor = sig_def.outputs['scales']
  scores_tensor = sig_def.outputs['scores']

  # Input image scales to use for extraction.
  image_scales = list(config.image_scales)

  # Custom configuration needed when local features are used.
  if config.use_local_features:
    score_threshold = config.delf_local_config.score_threshold
    max_feature_num = config.delf_local_config.max_feature_num

    # If using PCA, pre-load required parameters.
    local_pca_parameters = {}
    if config.delf_local_config.use_pca:
      local_pca_parameters['mean'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_local_config.pca_parameters.mean_path),
          dtype=tf.float32)
      local_pca_parameters['matrix'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_local_config.pca_parameters.projection_matrix_path),
          dtype=tf.float32)
      local_pca_parameters[
          'dim'] = config.delf_local_config.pca_parameters.pca_dim
      local_pca_parameters['use_whitening'] = (
          config.delf_local_config.pca_parameters.use_whitening)
      if config.delf_local_config.pca_parameters.use_whitening:
        local_pca_parameters['variances'] = tf.squeeze(
            tf.constant(
                datum_io.ReadFromFile(
                    config.delf_local_config.pca_parameters.pca_variances_path),
                dtype=tf.float32))
      else:
        local_pca_parameters['variances'] = None

  def ExtractorFn(image, resize_factor=1.0):
    """Receives an image and returns DELF global and/or local features.

    If image is too small, returns empty features.

    Args:
      image: Uint8 array with shape (height, width, 3) containing the RGB image.
      resize_factor: Optional float resize factor for the input image. If given,
        the maximum and minimum allowed image sizes in the config are scaled by
        this factor.

    Returns:
      extracted_features: A dict containing the extracted global descriptors
        (key 'global_descriptor' mapping to a [D] float array), and/or local
        features (key 'local_features' mapping to a dict with keys 'locations',
        'descriptors', 'scales', 'attention').
    """
    resized_image, scale_factors = utils.ResizeImage(
        image, config, resize_factor=resize_factor)

    # If the image is too small, returns empty features.
    if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[
        1] < _MIN_WIDTH:
      extracted_features = {'global_descriptor': np.array([])}
      if config.use_local_features:
        extracted_features.update({
            'local_features': {
                'locations': np.array([]),
                'descriptors': np.array([]),
                'scales': np.array([]),
                'attention': np.array([]),
            }
        })
      return extracted_features

    # Input tensors.
    image_tensor = tf.convert_to_tensor(resized_image)

    # Extracted features.
    extracted_features = {}
    output = None
    input_image_tensor = sig_def.inputs['input_image']
    input_scales_tensor = sig_def.inputs['input_scales']
    input_max_feature_num_tensor = sig_def.inputs['input_max_feature_num']
    iput_abs_thres_tensor = sig_def.inputs['iput_abs_thres']
    boxes_tensor = sig_def.outputs['boxes']
    features_tensor = sig_def.outputs['features']
    scales_tensor = sig_def.outputs['scales']
    scores_tensor = sig_def.outputs['scores']

    if config.use_local_features:
      feed_dict = {
              input_image_tensor: image_tensor,
              input_scales_tensor: image_scales_tensor
              }
      output = sess.run([image_tensor], feed_dict={input_image_tensor:image_tensor})

    # Post-process extracted features: normalize, PCA (optional), pooling.
    boxes = output[0]
    raw_local_descriptors = output[1]
    feature_scales = output[2]
    attention_with_extra_dim = output[3]

    attention = tf.reshape(attention_with_extra_dim,
                           [tf.shape(attention_with_extra_dim)[0]])
    locations, local_descriptors = (
        feature_extractor.DelfFeaturePostProcessing(
            boxes, raw_local_descriptors, config.delf_local_config.use_pca,
            local_pca_parameters))
    if not config.delf_local_config.use_resized_coordinates:
      locations /= scale_factors

    extracted_features.update({
        'local_features': {
            'locations': locations.numpy(),
            'descriptors': local_descriptors.numpy(),
            'scales': feature_scales.numpy(),
            'attention': attention.numpy(),
        }
    })

    return extracted_features

  return ExtractorFn

