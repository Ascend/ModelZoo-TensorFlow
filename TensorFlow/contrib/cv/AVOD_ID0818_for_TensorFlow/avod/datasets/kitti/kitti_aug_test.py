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


from npu_bridge.npu_init import *
import unittest

import numpy as np

from avod.datasets.kitti import kitti_aug


class KittiAugTest(unittest.TestCase):

    def test_flip_boxes_3d(self):

        boxes_3d = np.array([
            [1, 2, 3, 4, 5, 6, np.pi / 4],
            [1, 2, 3, 4, 5, 6, -np.pi / 4]
        ])

        exp_flipped_boxes_3d = np.array([
            [-1, 2, 3, 4, 5, 6, 3 * np.pi / 4],
            [-1, 2, 3, 4, 5, 6, -3 * np.pi / 4]
        ])

        flipped_boxes_3d = kitti_aug.flip_boxes_3d(boxes_3d)

        np.testing.assert_almost_equal(flipped_boxes_3d, exp_flipped_boxes_3d)

