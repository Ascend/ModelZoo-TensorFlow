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
import numpy as np
import unittest

from wavedata.tools.core import geometry_utils


class GeometryUtilsTest(unittest.TestCase):

    def test_dist_to_plane(self):

        xy_plane = [0, 0, 1, 0]
        xz_plane = [0, 1, 0, 0]
        yz_plane = [1, 0, 0, 0]
        diagonal_plane = [1, 1, 1, 0]

        point = [[1, 1, 1]]

        dist_from_xy = geometry_utils.dist_to_plane(xy_plane, point)
        dist_from_xz = geometry_utils.dist_to_plane(xz_plane, point)
        dist_from_yz = geometry_utils.dist_to_plane(yz_plane, point)
        dist_from_diag = geometry_utils.dist_to_plane(diagonal_plane, point)

        self.assertAlmostEqual(dist_from_xy[0], 1.0)
        self.assertAlmostEqual(dist_from_xz[0], 1.0)
        self.assertAlmostEqual(dist_from_yz[0], 1.0)
        self.assertAlmostEqual(dist_from_diag[0], np.sqrt(3))

        # Check that a signed distance is returned
        xy_plane_inv = [0, 0, -1, 0]
        diagonal_plane_inv = [-1, -1, -1, 0]

        dist_from_xy_inv = geometry_utils.dist_to_plane(xy_plane_inv, point)
        dist_from_diag_inv = geometry_utils.dist_to_plane(diagonal_plane_inv,
                                                          point)

        self.assertAlmostEqual(dist_from_xy_inv[0], -1.0)
        self.assertAlmostEqual(dist_from_diag_inv[0], -np.sqrt(3))


if __name__ == '__main__':
    unittest.main()

