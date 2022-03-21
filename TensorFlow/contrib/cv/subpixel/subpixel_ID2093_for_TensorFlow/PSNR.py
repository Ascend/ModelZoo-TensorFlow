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

from utils import checkimage, modcrop
import numpy as np
import math
import cv2
import glob
import os


def psnr_ycbcr(target, ref, scale):
    """
    to calculate psnr of ycbcr
    """
    target = np.array(target, dtype=np.float64)  # BGR2RGB
    target_data = rgb2ycbcr(target)  # need to conver a BGR Image to RGB
    # target_data = cv2.cvtColor(target, cv2.COLOR_BGR2YCR_CB) # BGR
    target_data = target_data[:, :, 0]

    ref = np.array(ref, dtype=np.float64)
    ref_data = rgb2ycbcr(ref)
    # ref_data = cv2.cvtColor(ref, cv2.COLOR_BGR2YCR_CB)
    ref_data = ref_data[:, :, 0]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255.0 / rmse)


def psnr(target, ref, scale):
    """
    to calculate psnr of rgb
    target: np.float64
    ref: np.float64
    return np.float64
    """
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255.0 / rmse)


def rgb2ycbcr(rgb_image):
    """
    convert rgb into ycbcr BGR
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float64)
    transform_matrix = np.array([[64.738, 129.057, 25.064],
                                 [-37.945, - 74.494, 112.439],
                                 [112.439, - 94.154, - 18.285]])
    transform_matrix = transform_matrix / 256.
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix

    return ycbcr_image
