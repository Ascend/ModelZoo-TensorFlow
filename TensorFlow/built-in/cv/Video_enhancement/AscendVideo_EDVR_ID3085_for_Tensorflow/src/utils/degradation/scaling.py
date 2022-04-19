# Copyright 2022 Huawei Technologies Co., Ltd
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
import cv2
import time
import random
from scipy.interpolate import interp2d

from src.utils.degradation.base import Base


class SpatialScaling(Base):
    """Up- and down-sampling degradation.

    Args:
        target_size: list[int], [W, H] specifying the size after resize.
        scales: list[float], [Sw, Sh] specifying the scales of each dimension.
    """
    def __init__(self, target_size, scales=None):
        # Note: target_size and scales should be in [x, y] or [w, h] order
        self.scales = scales
        self.target_size = target_size
        assert isinstance(self.target_size, tuple)


class NearestScaling(SpatialScaling):
    """Nearest scaling up and down.

    Args:
        kernel_width: int, kernel width to remedy the misalignment of nearest 
            sampling.
    """
    def __init__(self, target_size, scales, kernel_width):
        super(NearestScaling, self).__init__(target_size, scales)
        kernel = cv2.getGaussianKernel(21, kernel_width)
        self.kernel = kernel @ np.transpose(kernel)
        self.kernel = self.shift_pixels(self.kernel, scales)

    def shift_pixels(self, x, scales, upper_left=False):
        """shift pixel for super-resolution with different scale factors

        Args:
            x: WxHxC or WxH, image or kernel
            sf: scale factor
            upper_left: shift direction
        """
        h, w = x.shape[:2]
        shift = (scales-1)*0.5
        xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
        if upper_left:
            x1 = xv + shift[0]
            y1 = yv + shift[1]
        else:
            x1 = xv - shift[0]
            y1 = yv - shift[1]

        x1 = np.clip(x1, 0, w-1)
        y1 = np.clip(y1, 0, h-1)

        if x.ndim == 2:
            x = interp2d(xv, yv, x)(x1, y1)
        if x.ndim == 3:
            for i in range(x.shape[-1]):
                x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

        return x

    def apply(self, im):
        im = cv2.filter2D(im, -1, self.kernel)
        return cv2.resize(im, dsize=self.target_size, fx=0., fy=0., interpolation=cv2.INTER_NEAREST)


class BicubicScaling(SpatialScaling):
    """Bicubic sampling.
    """
    def apply(self, im):
        return cv2.resize(im, dsize=self.target_size, fx=0., fy=0., interpolation=cv2.INTER_LINEAR)


class BilinearScaling(SpatialScaling):
    """Bilinear sampling.
    """
    def apply(self, im):
        return cv2.resize(im, dsize=self.target_size, fx=0., fy=0., interpolation=cv2.INTER_CUBIC)
