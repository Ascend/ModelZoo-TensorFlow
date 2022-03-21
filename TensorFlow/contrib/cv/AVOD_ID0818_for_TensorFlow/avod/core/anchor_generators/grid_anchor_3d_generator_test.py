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


"""Grid Anchor Generation unit test module."""
from npu_bridge.npu_init import *
import unittest
import numpy as np

import avod.tests as tests

from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.builders.dataset_builder import DatasetBuilder


def generate_fake_dataset():
    return DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_UNITTEST)


class GridAnchor3dGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_kitti_dir = tests.test_path() + "/datasets/Kitti/object"
        cls.dataset = generate_fake_dataset()

        # create generic ground plane (normal vector is straight up)
        cls.ground_plane = np.array([0., -1., 0., 0.])
        cls.clusters = np.array([[1., 1., 1.], [2., 1., 1.]])

        cls.anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    def test_generate_anchors(self):
        normal_area = [(-1., 1.), (-1., 0.), (0., 1.)]
        no_x_area = [(0., 0.), (-1., 0.), (0., 2.)]
        no_z_area = [(-1., 1.), (-1., 0.), (0., 0.)]

        expected_anchors = np.array([[-0.5, 0., 0.5, 1., 1., 1., 0.],
                                     [-0.5, 0., 0.5, 1., 1., 1., np.pi / 2],
                                     [-0.5, 0., 0.5, 2., 1., 1., 0.],
                                     [-0.5, 0., 0.5, 2., 1., 1., np.pi / 2],
                                     [0.5, 0., 0.5, 1., 1., 1., 0.],
                                     [0.5, 0., 0.5, 1., 1., 1., np.pi / 2],
                                     [0.5, 0., 0.5, 2., 1., 1., 0.],
                                     [0.5, 0., 0.5, 2., 1., 1., np.pi / 2]])
        gen_anchors = \
            self.anchor_generator.generate(area_3d=normal_area,
                                           anchor_3d_sizes=self.clusters,
                                           anchor_stride=[1, 1],
                                           ground_plane=self.ground_plane)
        self.assertEqual(gen_anchors.shape, expected_anchors.shape)
        np.testing.assert_almost_equal(gen_anchors,
                                       expected_anchors,
                                       decimal=3)

        expected_anchors = np.ndarray(shape=(0, 7))
        gen_anchors = \
            self.anchor_generator.generate(area_3d=no_x_area,
                                           anchor_3d_sizes=self.clusters,
                                           anchor_stride=[1, 1],
                                           ground_plane=self.ground_plane)
        self.assertEqual(gen_anchors.shape, expected_anchors.shape)
        np.testing.assert_almost_equal(gen_anchors,
                                       expected_anchors,
                                       decimal=3)

        expected_anchors = np.ndarray(shape=(0, 7))
        gen_anchors = \
            self.anchor_generator.generate(area_3d=no_z_area,
                                           anchor_3d_sizes=self.clusters,
                                           anchor_stride=[1, 1],
                                           ground_plane=self.ground_plane)
        self.assertEqual(gen_anchors.shape, expected_anchors.shape)
        np.testing.assert_almost_equal(gen_anchors,
                                       expected_anchors,
                                       decimal=3)


if __name__ == '__main__':
    unittest.main()

