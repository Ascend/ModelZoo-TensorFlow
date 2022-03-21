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


"""A module for helper tensorflow ops."""
from npu_bridge.npu_init import *

import tensorflow as tf


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
    """Creates dense vector with indices set to specific value
       and rest to zeros.

      This function exists because it is unclear if it is safe to use
      tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
      with indices which are not ordered. This function accepts a
      dynamic size (e.g. tf.shape(tensor)[0])

    Args:
      indices: 1d Tensor with integer indices which are to be set to
               indices_values.
      size: scalar with size (integer) of output Tensor.
      indices_value: values of elements specified by indices in the output
                     vector
      default_value: values of other elements in the output vector.
      dtype: data type.

    Returns:
      dense 1D Tensor of shape [size] with indices set to indices_values and the
          rest set to default_value.
    """
    size = tf.to_int32(size)
    zeros = tf.ones([size], dtype=dtype) * default_value
    values = tf.ones_like(indices, dtype=dtype) * indices_value

    return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                             [zeros, values])

