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


""" 
Util functions implementing the camera

@@batch_orth_proj_idrot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def batch_orth_proj_idrot(X, camera, name=None):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """
    with tf.name_scope(name, "batch_orth_proj_idrot", [X, camera]):
        # TODO check X dim size.
        # tf.Assert(X.shape[2] == 3, [X])

        camera = tf.reshape(camera, [-1, 1, 3], name="cam_adj_shape")

        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)
