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


"""LabelClusterUtils unit test module"""
from npu_bridge.npu_init import *

import unittest

import array
import numpy as np
import os

import avod
import avod.tests as tests

from avod.builders.dataset_builder import DatasetBuilder
from avod.core.label_cluster_utils import LabelClusterUtils


class LabelClusterUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_kitti_dir = tests.test_path() + "/datasets/Kitti/object"
        cls.dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_UNITTEST)

    def test_get_clusters(self):

        # classes = ['Car', 'Pedestrian', 'Cyclist']
        num_clusters = [2, 1, 1]

        label_cluster_utils = LabelClusterUtils(self.dataset)
        clusters, std_devs = label_cluster_utils.get_clusters()

        # Check that correct number of clusters are returned
        clusters_per_class = [len(cls_clusters) for cls_clusters in clusters]
        std_devs_per_class = [len(cls_std_devs) for cls_std_devs in std_devs]

        self.assertEqual(clusters_per_class, num_clusters)
        self.assertEqual(std_devs_per_class, num_clusters)

        # Check that text files were saved
        txt_folder_exists = os.path.isdir(
            avod.root_dir() + "/data/label_clusters/unittest-kitti")
        self.assertTrue(txt_folder_exists)

        # Calling get_clusters again should read from files
        read_clusters, read_std_devs = label_cluster_utils.get_clusters()

        # Check that read values are the same as generated ones
        np.testing.assert_allclose(np.vstack(clusters),
                                   np.vstack(read_clusters))
        np.testing.assert_allclose(np.vstack(std_devs),
                                   np.vstack(read_std_devs))

    def test_flatten_data(self):
        data_to_reshape = list()

        data_to_reshape.append([[1, 2, 3], [4, 5, 6]])
        data_to_reshape.append([[7, 8, 9]])
        data_to_reshape.append([[10, 11, 12], [13, 14, 15]])

        expected_output = np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9],
                                    [10, 11, 12],
                                    [13, 14, 15]])

        label_cluster_utils = LabelClusterUtils(self.dataset)

        flattened = label_cluster_utils._flatten_data(data_to_reshape)
        np.testing.assert_array_equal(flattened,
                                      expected_output,
                                      err_msg='Wrong flattened array')

