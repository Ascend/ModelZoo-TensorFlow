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
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.color import rgb2ycbcr
import skimage.io as io


def ssim(image1, image2):
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image1 = rgb2ycbcr(image1)[:, :, 0:1]
    image2 = rgb2ycbcr(image2)[:, :, 0:1]
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    ssim_val = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True,
                                     data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
    return ssim_val



