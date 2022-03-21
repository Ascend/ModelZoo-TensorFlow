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


# --------------------------------------------------------
# Subcategory CNN
# Copyright (c) 2015 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from config import cfg


def get_boxes_grid(image_height, image_width):
  """
  Return the boxes on image grid.
  """

  # height and width of the heatmap
  if cfg.NET_NAME == 'CaffeNet':
    height = np.floor((image_height * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
    height = np.floor((height - 1) / 2.0 + 1 + 0.5)
    height = np.floor((height - 1) / 2.0 + 1 + 0.5)

    width = np.floor((image_width * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)
    width = np.floor((width - 1) / 2.0 + 1 + 0.5)
  elif cfg.NET_NAME == 'VGGnet':
    height = np.floor(image_height * max(cfg.TRAIN.SCALES) / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)
    height = np.floor(height / 2.0 + 0.5)

    width = np.floor(image_width * max(cfg.TRAIN.SCALES) / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
    width = np.floor(width / 2.0 + 0.5)
  else:
    assert (1), 'The network architecture is not supported in utils.get_boxes_grid!'

  # compute the grid box centers
  h = np.arange(height)
  w = np.arange(width)
  y, x = np.meshgrid(h, w, indexing='ij')
  centers = np.dstack((x, y))
  centers = np.reshape(centers, (-1, 2))
  num = centers.shape[0]

  # compute width and height of grid box
  area = cfg.TRAIN.KERNEL_SIZE * cfg.TRAIN.KERNEL_SIZE
  aspect = cfg.TRAIN.ASPECTS  # height / width
  num_aspect = len(aspect)
  widths = np.zeros((1, num_aspect), dtype=np.float32)
  heights = np.zeros((1, num_aspect), dtype=np.float32)
  for i in range(num_aspect):
    widths[0, i] = math.sqrt(area / aspect[i])
    heights[0, i] = widths[0, i] * aspect[i]

  # construct grid boxes
  centers = np.repeat(centers, num_aspect, axis=0)
  widths = np.tile(widths, num).transpose()
  heights = np.tile(heights, num).transpose()

  x1 = np.reshape(centers[:, 0], (-1, 1)) - widths * 0.5
  x2 = np.reshape(centers[:, 0], (-1, 1)) + widths * 0.5
  y1 = np.reshape(centers[:, 1], (-1, 1)) - heights * 0.5
  y2 = np.reshape(centers[:, 1], (-1, 1)) + heights * 0.5

  boxes_grid = np.hstack((x1, y1, x2, y2)) / cfg.TRAIN.SPATIAL_SCALE

  return boxes_grid, centers[:, 0], centers[:, 1]
