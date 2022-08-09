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


def preprocess_img(config, image, resize_factor=1.0):
  # Assert the configuration.
  if not config.use_local_features and not config.use_global_features:
    raise ValueError('Invalid config: at least one of '
                     '{use_local_features, use_global_features} must be True')

  # Input image scales to use for extraction.
  resized_image, scale_factors = utils.ResizeImage(image, config, resize_factor=resize_factor)

  # If the image is too small, returns empty features.
  if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[1] < _MIN_WIDTH:
    return None
  else:
    return (resized_image, scale_factors)


def extrator_fn(config, boxes, features, scores, scale_factors):
  """Creates a function to extract global and/or local features from an image.

  Args:
    config: DelfConfig proto containing the model configuration.

  Returns:
    Function that receives an image and returns features.

  Raises:
    ValueError: if config is invalid.
  """
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

  # Post-process extracted features: normalize, PCA (optional), pooling.
  attention = tf.reshape(scores, [tf.shape(scores)[0]])
  locations, local_descriptors = (
      feature_extractor.DelfFeaturePostProcessing(
          boxes, features, config.delf_local_config.use_pca,
          local_pca_parameters))
  if not config.delf_local_config.use_resized_coordinates:
    locations /= scale_factors
  return locations, local_descriptors, attention 

