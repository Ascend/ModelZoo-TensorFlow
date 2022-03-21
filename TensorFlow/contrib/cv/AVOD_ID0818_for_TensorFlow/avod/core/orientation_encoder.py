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
import tensorflow as tf


def tf_orientation_to_angle_vector(orientations_tensor):
    """ Converts orientation angles into angle unit vector representation.
        e.g. 45 -> [0.717, 0.717], 90 -> [0, 1]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 2) of angle unit vectors in the format [x, y]
    """
    x = tf.cos(orientations_tensor)
    y = tf.sin(orientations_tensor)

    return tf.stack([x, y], axis=1)


def tf_angle_vector_to_orientation(angle_vectors_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.717, 0.717] -> 45, [0, 1] -> 90

    Args:
        angle_vectors_tensor: a tensor of shape (N, 2) of angle unit vectors
            in the format [x, y]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    x = angle_vectors_tensor[:, 0]
    y = angle_vectors_tensor[:, 1]

    return tf.atan2(y, x)

