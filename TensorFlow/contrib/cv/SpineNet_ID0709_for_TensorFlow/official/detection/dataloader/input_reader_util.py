# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utility function for input reader."""
from npu_bridge.npu_init import *

import tensorflow.compat.v1 as tf

from ops import spatial_transform_ops


def transform_image_for_tpu(batch_images,
                            space_to_depth_block_size=1,
                            transpose_images=True):
  """Transforms batched images to optimize memory usage on TPU.

  Args:
    batch_images: Batched images in the shape [batch_size, image_height,
      image_width, num_channel].
    space_to_depth_block_size: An integer for space-to-depth block size. The
      input image's height and width must be divisible by block_size. The block
      size also needs to match the stride length of the first conv layer. See
      go/auto-space-to-depth and tf.nn.space_to_depth.
    transpose_images: Whether or not transpose image dimensions.

  Returns:
    transformed batched images.
  """
  if space_to_depth_block_size > 1:
    return spatial_transform_ops.fused_transpose_and_space_to_depth(
        batch_images, space_to_depth_block_size, transpose_images)
  elif transpose_images:
    # Transpose the input images from [N,H,W,C] to [H,W,C,N] since reshape on
    # TPU is expensive.
    return tf.transpose(batch_images, [1, 2, 3, 0])
  else:
    return batch_images

