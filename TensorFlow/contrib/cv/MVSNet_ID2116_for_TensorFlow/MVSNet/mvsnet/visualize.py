#!/usr/bin/env python
#
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
"""
Copyright 2019, Yao Yao, HKUST.
Depth map visualization.
"""
from npu_bridge.npu_init import *

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from preprocess import load_pfm
from depthfusion import read_gipuma_dmb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    depth_path = args.depth_path
    if depth_path.endswith('npy'):
        depth_image = np.load(depth_path)
        depth_image = np.squeeze(depth_image)
        print('value range: ', depth_image.min(), depth_image.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('pfm'):
        depth_image = load_pfm(open(depth_path, 'rb'))
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    elif depth_path.endswith('dmb'):
        depth_image = read_gipuma_dmb(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image, 'rainbow')
        plt.show()
    else:
        depth_image = cv2.imread(depth_path)
        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
        print('value range: ', ma.min(), ma.max())
        plt.imshow(depth_image)
        plt.show()

