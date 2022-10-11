#!/usr/bin/python
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements fully-supervised model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from npu_bridge.npu_init import *

import tensorflow as tf

from models.utils import get_net

def model_fn(features, mode):
  """
  Args:
    features: Dict of inputs containing, among others, "image" and "label."
    mode: model's mode: training, eval or prediction

  Returns:
    EstimatorSpec
  """

  images = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    with tf.variable_scope('module'):
        net = get_net(num_classes=1000)
        logits, end_points = net(images, False)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)