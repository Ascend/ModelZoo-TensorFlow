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

from wavedata.tools.core.integral_image_2d import IntegralImage2D


class IntegralImage2DTest(unittest.TestCase):

    def test_integral_image_2d(self):

        test_mat = np.ones((3, 3)).astype(np.float32)

        # Generate integral image
        integral_image = IntegralImage2D(test_mat)
        boxes = np.array([[0, 0, 1, 1],
                          [0, 0, 2, 2],
                          [0, 0, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(boxes)

        # First box case = should be 1*1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)
        # Second box case = should be 2*2*2 = 8
        self.assertTrue(occupancy_count[1] == 4)
        # Third box case = should be 3*3*3 = 27
        self.assertTrue(occupancy_count[2] == 9)

        boxes = np.array([[1, 1, 2, 2],
                          [1, 1, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(boxes)

        # First box case = should be 1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)

        # Second box case = should be 2*2 = 4
        self.assertTrue(occupancy_count[1] == 4)

        boxes = np.array([[0, 0, 3, 1]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(boxes)

        # Flat Surface case = should be 1*3 = 3
        self.assertTrue(occupancy_count[0] == 3)

        # Test outside the boundary
        boxes = np.array([[0, 0, 2312, 162]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(boxes)
        self.assertTrue(occupancy_count[0] == 9)


if __name__ == '__main__':
    unittest.main()

