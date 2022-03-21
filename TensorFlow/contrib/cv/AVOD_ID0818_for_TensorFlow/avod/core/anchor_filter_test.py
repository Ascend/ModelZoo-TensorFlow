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

from avod.core import anchor_filter
from avod.core import box_3d_encoder
from wavedata.tools.core.voxel_grid import VoxelGrid


class AnchorFilterTest(unittest.TestCase):

    def test_get_empty_anchor_filter_in_2d(self):
        # create generic ground plane (normal vector is straight up)
        area_extent = [(0., 2.), (-1., 0.), (0., 2.)]

        # Creates a voxel grid in following format at y = bin (-1.5, -0.5]
        # [ ][ ][ ][ ]
        # [ ][ ][x][ ]
        # [ ][ ][ ][ ]
        # [ ][ ][x][ ]
        pts = np.array([[0.51, -0.5, 1.1],
                        [1.51, -0.5, 1.1]])

        voxel_size = 0.5
        voxel_grid = VoxelGrid()
        voxel_grid.voxelize(pts, voxel_size, extents=area_extent)

        # Define anchors to test
        boxes_3d = np.array([
            [0.51, 0, 0.51, 1, 1, 1, 0],
            [0.51, 0, 0.51, 1, 1, 1, np.pi / 2.],
            [0.51, 0, 1.1, 1, 1, 1, 0],
            [0.51, 0, 1.1, 1, 1, 1, np.pi / 2.],
            [1.51, 0, 0.51, 1, 1, 1, 0],
            [1.51, 0, 0.51, 1, 1, 1, np.pi / 2.],
            [1.51, 0, 1.1, 1, 1, 1, 0],
            [1.51, 0, 1.1, 1, 1, 1, np.pi / 2.],
        ])

        anchors = box_3d_encoder.box_3d_to_anchor(boxes_3d)

        # test anchor locations, number indicates the anchors indices
        # [ ][ ][ ][ ]
        # [ ][1][3][ ]
        # [ ][ ][ ][ ]
        # [ ][5][7][ ]

        gen_filter = anchor_filter.get_empty_anchor_filter(anchors,
                                                           voxel_grid,
                                                           density_threshold=1)

        expected_filter = np.array(
            [False, False, True, True, False, False, True, True])

        self.assertTrue((gen_filter == expected_filter).all())

        boxes_3d = np.array([
            [0.5, 0, 0.5, 2, 1, 1, 0],  # case 1
            [0.5, 0, 0.5, 2, 1, 1, np.pi / 2.],
            [0.5, 0, 1.5, 1, 2, 1, 0],  # case 2
            [0.5, 0, 1.5, 1, 2, 1, np.pi / 2.],
            [1.5, 0, 0.5, 2, 1, 1, 0],  # case 3
            [1.5, 0, 0.5, 2, 1, 1, np.pi / 2.],
            [1.5, 0, 1.5, 1, 2, 1, 0],  # case 4
            [1.5, 0, 1.5, 1, 2, 1, np.pi / 2.]
        ])

        anchors = box_3d_encoder.box_3d_to_anchor(boxes_3d)

        # case 1
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][o][ ][ ]   [ ][o][o][ ]
        # [ ][o][ ][ ]   [ ][ ][ ][ ]
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]

        # case 2
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][ ][o][o]   [ ][ ][o][ ]
        # [ ][ ][ ][ ]   [ ][ ][o][ ]
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]

        # case 3
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][o][ ][ ]   [ ][o][o][ ]
        # [ ][o][ ][ ]   [ ][ ][ ][ ]

        # case 4
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][ ][ ][ ]   [ ][ ][ ][ ]
        # [ ][ ][o][o]   [ ][ ][o][ ]
        # [ ][ ][ ][ ]   [ ][ ][o][ ]

        gen_filter = anchor_filter.get_empty_anchor_filter(anchors,
                                                           voxel_grid,
                                                           density_threshold=1)
        expected_filter = np.array(
            [False, True, True, True, False, True, True, True])

        self.assertTrue((gen_filter == expected_filter).all())


if __name__ == '__main__':
    unittest.main()

