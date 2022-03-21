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


""" This module provides an interface to calculate values inside a box given
    a set of coordinates and an image """
from npu_bridge.npu_init import *
import numpy as np


class IntegralImage2D(object):
    def __init__(self, img):
        # initialize with the 2D integral image
        self._integral_image = self._integral_image_2d(img)

        # keep map sizes
        self._x_size = self._integral_image.shape[0]
        self._z_size = self._integral_image.shape[1]

    def _integral_image_2d(self, img):
        """Calculates a 2D integral image from an input image.

        :param img :    W x L array
                        Integral image of size W x L

        :return rt_image : IntegralImage object
                           Object containing integral image and its parameters.
                           Returns empty list on failure.
        """
        # Check if points are 2D, otherwise early exit
        if img.ndim != 2:
            raise ValueError('Not a 2D image for integral image: input dim {}'
                             .format(img.ndim))

        integral_image = np.cumsum(np.cumsum(img, 0), 1)

        # pad integral image with 0s on one side of each dimension required
        # so that when accessing coordinate n-1, we get a valid value of 0
        integral_image_out = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
        integral_image_out[1:, 1:] = integral_image

        return integral_image_out

    def query(self, boxes):
        """Input is an array of 2D boxes 4 coordinates. Each column represents
        a cuboid in the format [x1, z1, x2, z2].T. Thus, the dimensions
        should be 4 x N. The 2 sets of 2D coordinates represent the 2 corners of
        the bounding box. The first set of coordinates is the point closest to
        the origin of the image. The second set of coordinates is the point
        farthest from the origin.

        :param boxes : 4 x N ndarray
                       Contains the (x1, z1) and (x2, z2) coordinates
                       of the box to query.

        :return param : N x 1 ndarray
                        List consists of values contained inside box specified
                        by coordinates from boxes. Empty on failure.
        """
        boxes = np.asarray(boxes)

        # check size
        if boxes.shape[0] != 4:
            raise ValueError('Incorrect number of dimensions for query: '
                             'input dim {}'.format(boxes.shape[0]))

        if boxes.shape[1] < 1:
            raise ValueError('The dimension N must be an integer greater than '
                             '1: input dim {}'.format(boxes.shape[1]))

        if boxes.dtype != np.uint32:
            raise TypeError('boxes must be type of np.uint32')

        # Clip all the maximum coordinates to the voxelgrid size
        # Note: The integral image gets zero padded.
        max_extents = np.array([self._x_size, self._z_size,
                                self._x_size, self._z_size]) - 1

        # Make sure all boxes are within the maximum extents
        boxes = np.minimum(boxes, max_extents.reshape(4, -1)).astype(np.uint32)

        x1 = boxes[0, :]
        z1 = boxes[1, :]
        x2 = boxes[2, :]
        z2 = boxes[3, :]

        output = self._integral_image[x2, z2] + \
            self._integral_image[x1, z1] - \
            self._integral_image[x2, z1] - \
            self._integral_image[x1, z2]

        return output

