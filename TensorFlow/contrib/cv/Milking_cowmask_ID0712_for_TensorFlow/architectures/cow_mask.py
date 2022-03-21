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

"""Cow mask generation."""
from npu_bridge.npu_init import *
import math
import numpy as np
import tensorflow as tf
from architectures import tf_conv_general
_ROOT_2 = math.sqrt(2.0)
_ROOT_2_PI = math.sqrt(2.0 * math.pi)


def gaussian_kernels(sigmas, max_sigma):
  """Make Gaussian kernels for Gaussian blur.

  Args:
      sigmas: kernel sigmas as a [N] jax.numpy array
      max_sigma: sigma upper limit as a float (this is used to determine
        the size of kernel required to fit all kernels)

  Returns:
      a (N, kernel_width) array
  """
  sigmas = sigmas[:, None]
  size = round(max_sigma * 3) * 2 + 1
  x = np.arange(-size, size + 1)[None, :].astype(np.float32)
  y = tf.exp(-0.5 * x ** 2 / sigmas ** 2)
  return y / (sigmas * _ROOT_2_PI)


def cow_masks(n_masks, mask_size, log_sigma_range, max_sigma,
              prop_range, seed):
  """Generate Cow Mask.

  Args:
      n_masks: number of masks to generate as an int
      mask_size: image size as a `(height, width)` tuple
      log_sigma_range: the range of the sigma (smoothing kernel)
          parameter in log-space`(log(sigma_min), log(sigma_max))`
      max_sigma: smoothing sigma upper limit
      prop_range: range from which to draw the proportion `p` that
        controls the proportion of pixel in a mask that are 1 vs 0
  Returns:
      Cow Masks shaped [v, height, width, 1]
  """
  # rng_k1, rng_k2 = jax.random.split(rng_key)
  # rng_k2, rng_k3 = jax.random.split(rng_k2)

  # Draw the per-mask proportion p

  p = tf.random.uniform((n_masks,),
      prop_range[0], prop_range[1],seed=seed)
  # Compute threshold factors
  y = 2 * p - 1
  threshold_factors = tf.math.reciprocal(tf.math.erf(y)) * _ROOT_2

  sigmas = tf.exp(tf.random.uniform((n_masks,), log_sigma_range[0], log_sigma_range[1], seed=seed))

  # Create initial noise with the batch and channel axes swapped so we can use
  # tf.nn.depthwise_conv2d to convolve it with the Gaussian kernels
  noise = tf.random.normal((1,) + mask_size + (n_masks,), seed=seed)

  # Generate a kernel for each sigma
  kernels = gaussian_kernels(sigmas, max_sigma)
  # kernels: [batch, width] -> [width, batch]
  kernels = tf.transpose(kernels,(1, 0))
  # kernels in y and x
  krn_y = kernels[:, None, None, :]
  krn_x = kernels[None, :, None, :]

  # Apply kernels in y and x separately

  smooth_noise = tf_conv_general.conv_general_dilated(
      noise, krn_y, 1, 'SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=n_masks)
  smooth_noise = tf_conv_general.conv_general_dilated(
      smooth_noise, krn_x, (1, 1), 'SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=n_masks)

  # [1, height, width, batch] -> [batch, height, width, 1]
  smooth_noise = tf.transpose(smooth_noise, (3, 1, 2, 0))

  # Compute mean and std-dev
  noise_mu, noise_sigma = tf.nn.moments(smooth_noise,axes=(1, 2, 3), keepdims=True)
  # Compute thresholds
  thresholds = threshold_factors[:, None, None, None] * noise_sigma + noise_mu
  # Apply threshold
  masks = (smooth_noise <= thresholds)
  masks = tf.cast(masks, tf.float32)
  return masks

