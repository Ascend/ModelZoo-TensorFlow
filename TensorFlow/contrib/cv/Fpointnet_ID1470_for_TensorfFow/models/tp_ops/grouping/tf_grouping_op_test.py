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

from __future__ import print_function
import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, group_point

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:0'):
      points = tf.constant(np.random.random((1,128,16)).astype('float32'))
      print(points)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      radius = 0.3 
      nsample = 32
      idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
      grouped_points = group_point(points, idx)
      print(grouped_points)

    with self.test_session():
      print("---- Going to compute gradient error")
      err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
      print(err)
      self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
