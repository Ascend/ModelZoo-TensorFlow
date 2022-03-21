# Copyright 2017 Phil Ferriere. All Rights Reserved.
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
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
correlation.py

Computes cross correlation between two feature maps.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py
        Written by Patrick Wieschollek, Copyright Yuxin Wu
        Apache License 2.0
"""

from __future__ import absolute_import, division, print_function
from npu_bridge.npu_init import *
import tensorflow as tf


def correlation(c1, c2, search_range, name='corr_layer'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: input image1
        c2: input image2
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(c2, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    corr = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            corr.append(cost)
    corr = tf.concat(corr, axis=3)
    corr = tf.nn.leaky_relu(corr, alpha=0.1, name=name)

    return corr
