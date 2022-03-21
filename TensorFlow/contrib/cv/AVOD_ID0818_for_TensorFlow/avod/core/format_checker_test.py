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
import tensorflow as tf

from avod.core import format_checker as fc
from wavedata.tools.obj_detection import obj_utils


class FormatCheckerTest(unittest.TestCase):

    def test_check_box_3d_format(self):

        # Case 1, invalid type
        test_var = [0, 0, 0, 0, 0, 0, 0]
        np.testing.assert_raises(TypeError,
                                 fc.check_box_3d_format, test_var)

        # Case 2, invalid shape
        test_var = np.ones([1, 5])
        np.testing.assert_raises(TypeError,
                                 fc.check_box_3d_format, test_var)

        test_var = np.ones([5, 6])
        np.testing.assert_raises(TypeError,
                                 fc.check_box_3d_format, test_var)

        test_var = np.ones([1, 7])
        fc.check_box_3d_format(test_var)

        test_var = np.ones([10, 7])
        fc.check_box_3d_format(test_var)

        test_var = tf.ones([5, 7])
        fc.check_box_3d_format(test_var)

        test_var = tf.ones([5, 3])
        np.testing.assert_raises(TypeError,
                                 fc.check_box_3d_format, test_var)

    def test_check_object_label_format(self):
        test_obj = obj_utils.ObjectLabel()
        test_obj.h = 1
        test_obj.w = 1
        test_obj.l = 1
        test_obj.t = [1, 1, 1]
        test_obj.ry = 0

        # Case 1, Single instance of object label
        test_obj_list = [test_obj]
        fc.check_object_label_format(test_obj_list)

        test_obj_list = [test_obj, test_obj, test_obj]
        fc.check_object_label_format(test_obj_list)

        test_obj_list = [test_obj, test_obj, '0']
        np.testing.assert_raises(TypeError,
                                 fc.check_object_label_format, test_obj_list)

        # Case 2, Range check
        test_obj.t = [1, 1]
        test_obj_list = [test_obj]
        np.testing.assert_raises(TypeError,
                                 fc.check_object_label_format, test_obj_list)

    def test_check_anchor_format(self):
        # Case 1, invalid type
        test_var = [0, 0, 0, 0, 0, 0]
        np.testing.assert_raises(TypeError,
                                 fc.check_anchor_format, test_var)

        # Case 2, invalid shape
        test_var = np.ones([1, 5])
        np.testing.assert_raises(TypeError,
                                 fc.check_anchor_format, test_var)

        test_var = np.ones([1, 6])
        fc.check_anchor_format(test_var)

        test_var = np.ones([5, 6])
        fc.check_anchor_format(test_var)

        test_var = tf.ones([5, 6])
        fc.check_anchor_format(test_var)

        test_var = tf.ones([5, 4])
        np.testing.assert_raises(TypeError,
                                 fc.check_anchor_format, test_var)

