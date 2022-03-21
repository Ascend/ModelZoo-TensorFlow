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

# """The data base wrapper class"""
from npu_bridge.npu_init import *

import os
import random
import shutil

import numpy as np

from utils.util import *


class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._image_set = []
    self._image_idx = []
    self._data_root_path =[]
    self.mc = mc
    # batch reader
    self._perm_idx = []
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0
  def parse_function(self, record):
    mc = self.mc
    if mc.DATA_AUGMENTATION:
      if mc.RANDOM_FLIPPING:
        if np.random.rand() > 0.5:
          # flip y
          record = record[:, ::-1, :]
          record[:, :, 1] *= -1
    lidar = record[:, :, :5]  # x, y, z, intensity, depth
    lidar_mask = np.reshape((lidar[:, :, 4] > 0), [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])
    # normalize
    lidar = (lidar - mc.INPUT_MEAN) / mc.INPUT_STD
    label = record[:, :, 5]
    weight = np.zeros(label.shape)
    for l in range(mc.NUM_CLASS):
      weight[label == l] = mc.CLS_LOSS_WEIGHT[int(l)]
    return lidar, lidar_mask, label, weight

  def read_npy_file(self, idx):
    data = np.load(self._lidar_2d_path_at(idx.decode().split('/')[-1].split('.')[0]))
    return data.astype(np.float32)

  def data_preprocessing(self,data):
    lidar, lidar_mask, label, weight = self.parse_function(data)
    return lidar.astype(np.float32), lidar_mask.astype(np.float32), label.astype(np.int32), weight.astype(np.float32)

  def read_npydata(self, filename):
    mc = self.mc
    [npydata, ] = tf.py_func(self.read_npy_file, [filename], [tf.float32, ])
    lidar, lidar_mask, label, weight = tf.py_func(self.data_preprocessing, [npydata],
                                                  [tf.float32, tf.float32, tf.int32, tf.float32])
    lidar = tf.reshape(lidar, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 5])
    lidar_mask = tf.reshape(lidar_mask, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])
    label = tf.reshape(label, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL])
    weight = tf.reshape(weight, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL])
    return lidar, lidar_mask, label, weight

  def read_batch(self):
    mc = self.mc
    dataset = tf.data.Dataset.list_files(self._image_idx).repeat()
    dataset = dataset.map(lambda value: self.read_npydata(value))
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    dataset = dataset.batch(mc.BATCH_SIZE, drop_remainder=True)
    data_iterator = dataset.make_initializable_iterator()
    return data_iterator

  def evaluate_detections(self):
    raise NotImplementedError
