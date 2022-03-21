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
# limitations under the License.import os
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

import numpy as np
import cv2
import time


def gen_mask(flags):
    np.random.seed(seed=int(time.time()))  # set random seed according to the current time
    masks = np.ones((flags.sample_batch, flags.img_size, flags.img_size), dtype=np.float32)

    if flags.mask_type == 'center':
        scale = 0.25
        low, upper = int(flags.img_size * scale), int(flags.img_size * (1.0 - scale))
        masks[:, low:upper, low:upper] = 0.
    elif flags.mask_type == 'random':
        ratio = 0.8
        masks[np.random.random((flags.sample_batch, flags.img_size, flags.img_size)) <= ratio] = 0.
    elif flags.mask_type == 'half':
        half_types = np.random.randint(4, size=flags.sample_batch)
        masks = [half_mask(half_types[idx], flags.img_size) for idx in range(flags.sample_batch)]
        masks = np.asarray(masks)
    elif flags.mask_type == 'pattern':
        masks = [pattern_mask(flags.img_size) for _ in range(flags.sample_batch)]
        masks = np.asarray(masks)
    else:
        raise NotImplementedError

    return masks


def half_mask(half_type, img_size):
    mask = np.ones((img_size, img_size), dtype=np.float32)
    half = int(img_size / 2.)

    if half_type == 0:  # top mask
        mask[:half, :] = 0.
    elif half_type == 1:  # bottom mask
        mask[half:, :] = 0.
    elif half_type == 2:  # left mask
        mask[:, :half] = 0.
    elif half_type == 3:  # right mask
        mask[:, half:] = 0.
    else:
        raise NotImplementedError

    return mask


def pattern_mask(img_size):
    num_points, ratio = 3, 0.25
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for num in range(num_points):
        coordinate = np.random.randint(img_size, size=2)
        mask[coordinate[0], coordinate[1]] = 1.
        mask = cv2.dilate(mask, kernel, iterations=1)

    while np.sum(mask) < ratio * img_size * img_size:
        flag = True
        while flag:
            coordinate = np.random.randint(img_size, size=2)
            if mask[coordinate[0], coordinate[1]] == 1.:
                mask2 = np.zeros((img_size, img_size), dtype=np.float32)
                mask2[coordinate[0], coordinate[1]] = 1.
                mask2 = cv2.dilate(mask2, kernel, iterations=1)

                mask[mask + mask2 >= 1.] = 1.
                flag = False

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return 1. - mask

