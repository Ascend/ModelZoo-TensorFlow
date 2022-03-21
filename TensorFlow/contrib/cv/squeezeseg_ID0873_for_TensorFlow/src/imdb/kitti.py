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
# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017

"""Image data base class for kitti"""
from npu_bridge.npu_init import *

import os
import numpy as np
import subprocess

from imdb.imdb import imdb


class kitti(imdb):
  def __init__(self, image_set, data_path, mc):
    imdb.__init__(self, 'kitti_'+image_set, mc)
    self._image_set = image_set
    self._data_root_path = data_path
    self._lidar_2d_path = os.path.join(self._data_root_path, 'lidar_2d')
    self._gta_2d_path = os.path.join(self._data_root_path, 'gta')

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx()
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(
        self._data_root_path, 'ImageSet', self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file,"r") as f:
      image_idx = [self._lidar_2d_path_at(x.strip()) for x in f.readlines()]
    return image_idx

  def _lidar_2d_path_at(self, idx):
    if idx[:4] == 'gta_':
      lidar_2d_path = os.path.join(self._gta_2d_path, idx+'.npy')
    else:
      lidar_2d_path = os.path.join(self._lidar_2d_path, idx+'.npy')

    assert os.path.exists(lidar_2d_path), \
        'File does not exist: {}'.format(lidar_2d_path)
    return lidar_2d_path