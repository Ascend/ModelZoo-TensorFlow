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

def get_neighbours(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    return x >=0 and x < w and y >= 0 and y < h

def decode_image_by_join(pixel_scores, link_scores, pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    done_mask = np.zeros(pixel_mask.shape, np.bool)
    result_mask = np.zeros(pixel_mask.shape, np.int32)
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_id = 0
    for point in points:
        if done_mask[point]:
            continue
        group_id += 1
        group_q = [point]
        result_mask[point] = group_id
        while len(group_q):
            y, x = group_q[-1]
            group_q.pop()
            if not done_mask[y,x]:
                done_mask[y,x], result_mask[y,x] = True, group_id
                for n_idx, (nx, ny) in enumerate(get_neighbours(x, y)):
                    if is_valid_cord(nx, ny, w, h) and pixel_mask[ny, nx] and (link_mask[y, x, n_idx] or link_mask[ny, nx, 7 - n_idx]):
                        group_q.append((ny, nx))
    return result_mask
