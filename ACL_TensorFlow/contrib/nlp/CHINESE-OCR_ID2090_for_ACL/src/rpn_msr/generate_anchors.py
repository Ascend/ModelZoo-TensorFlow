# encoding:utf-8
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
from npu_bridge.npu_init import *
import numpy as np


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed

    embed()
