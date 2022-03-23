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
import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

def create_loader():
  return BaseLoader()

class BaseLoader():

  def __init__(self):
    pass

  def prepare(self):
    """
    Prepare the data loader.
    """
    raise NotImplementedError
  
  def get_num_images(self):
    """
    Get the number of images.
    Returns:
      The number of images.
    """
    raise NotImplementedError
  
  def get_patch_batch(self, batch_size, scale, input_patch_size):
    """
    Get a batch of input and ground-truth image patches.
    Args:
      batch_size: The number of image patches.
      scale: Scale of the input image patches.
      input_patch_size: Size of the input image patches.
    Returns:
      input_list: A list of input image patches.
      truth_list: A list of ground-truth image patches.
    """
    raise NotImplementedError
  
  def get_random_image_patch_pair(self, scale, input_patch_size):
    """
    Get a random pair of input and ground-truth image patches.
    Args:
      scale: Scale of the input image patch.
      input_patch_size: Size of the input image patch.
    Returns:
      A tuple of (input image patch, ground-truth image patch).
    """
    raise NotImplementedError

  def get_image_patch_pair(self, image_index, scale, input_patch_size):
    """
    Get a pair of input and ground-truth image patches for given image index.
    Args:
      image_index: Index of the image to be retrieved. Should be in [0, get_num_images()-1].
      scale: Scale of the input image patch.
      input_patch_size: Size of the input image patch.
    Returns:
      A tuple of (input image patch, ground-truth image patch).
    """
    raise NotImplementedError
  
  def get_image_pair(self, image_index, scale):
    """
    Get a pair of input and ground-truth images for given image index.
    Args:
      image_index: Index of the image to be retrieved. Should be in [0, get_num_images()-1].
      scale: Scale of the input image.
    Returns:
      A tuple of (input image, ground-truth image, image_name).
    """
    raise NotImplementedError