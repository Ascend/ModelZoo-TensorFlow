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

from avod.core import box_3d_projector


class Box3dProjectorTest(unittest.TestCase):
    def test_project_to_bev(self):
        boxes_3d = np.array([[0, 0, 0, 1, 0.5, 1, 0],
                             [0, 0, 0, 1, 0.5, 1, np.pi / 2],
                             [1, 0, 1, 1, 0.5, 1, np.pi / 2]])

        box_points, box_points_norm = \
            box_3d_projector.project_to_bev(boxes_3d, [[-1, 1], [-1, 1]])

        expected_boxes = np.array(
            [[[0.5, 0.25],
              [-0.5, 0.25],
              [-0.5, -0.25],
              [0.5, -0.25]],
             [[0.25, -0.5],
              [0.25, 0.5],
              [-0.25, 0.5],
              [-0.25, -0.5]],
             [[1.25, 0.5],
              [1.25, 1.5],
              [0.75, 1.5],
              [0.75, 0.5]]],
            dtype=np.float32)

        for box, exp_box in zip(box_points, expected_boxes):
            np.testing.assert_allclose(box, exp_box, rtol=1E-5)

