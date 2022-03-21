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

"""Regularizer classes for semi-supervised image classification."""
from npu_bridge.npu_init import *
import abc
import math
from architectures import cow_mask
import tensorflow as tf
import numpy as np


class Regularizer(object):
  """Abstract base regularizer."""

  @abc.abstractmethod
  def perturb_sample(self, image_batch, seed):
    pass

  @abc.abstractmethod
  def mix_images(self, image0_batch, image1_batch, seed):
    pass

  @abc.abstractmethod
  def mix_samples(self, image0_batch, y0_batch, image1_batch, y1_batch,
                  rng_key):
    image_batch, blend_factors = self.mix_images(
        image0_batch, image1_batch, rng_key)
    y_batch = y0_batch + (y1_batch - y0_batch) * blend_factors[:, None]
    return image_batch, y_batch, blend_factors

class CowMaskRegularizer(Regularizer):
  """CowMask regularizer."""

  def __init__(self, backg_noise_std, mask_prob, cow_sigma_range,
               cow_prop_range):
    self.backg_noise_std = backg_noise_std
    self.mask_prob = mask_prob
    self.cow_sigma_range = cow_sigma_range
    self.cow_prop_range = cow_prop_range
    self.log_sigma_range = (math.log(cow_sigma_range[0]),
                            math.log(cow_sigma_range[1]))
    self.max_sigma = cow_sigma_range[1]

  def perturb_sample(self, image_batch, seed):
    mask_size = image_batch.shape[1:3]
    masks = cow_mask.cow_masks(
        image_batch.get_shape()[0], mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, seed=seed)
    if self.mask_prob < 1.0:
      b = tf.random.bernoulli(self.mask_prob,
                               shape=(image_batch.get_shape()[0], 1, 1, 1), seed=seed)
      b = b.astype(np.float32)
      masks = 1.0 + (masks - 1.0) * b
    if self.backg_noise_std > 0.0:
      noise = tf.random.normal(image_batch.shape, seed=seed) * \
          self.backg_noise_std
      return image_batch * masks + noise * (1.0 - masks)
    else:
      return image_batch * masks

  def mix_images(self, image0_batch, image1_batch, seed):
    n_samples = image0_batch.get_shape()[0]
    mask_size = image0_batch.get_shape()[1:3]
    masks = cow_mask.cow_masks(
        n_samples, mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, seed=seed)
    blend_factors = tf.reduce_mean(masks, axis=(1, 2, 3))
    image_batch = image0_batch + (image1_batch - image0_batch) * masks
    return image_batch, blend_factors

