# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
This file contains a general conv operation for TensorFlow.
Zhibo Zhang, 2020.06.06
"""
from npu_bridge.npu_init import *
import tensorflow.nn as nn

def _conv_general_param_type_converter(dim, window_strides, lhs_dilation, rhs_dilation):
  """ Convert the inputs strides, lhs_dilation, rhs_dilation to the standard
  TF conv inputs."""
  strides = [window_strides] * dim if isinstance(window_strides, int) else \
            list(window_strides)
  if lhs_dilation:
    lhs_dilation = [lhs_dilation] * dim if isinstance(lhs_dilation, int) else \
                    list(lhs_dilation)
  if rhs_dilation:
    rhs_dilation = [rhs_dilation] * dim if isinstance(rhs_dilation, int) else \
                    list(rhs_dilation)
  return (strides, lhs_dilation, rhs_dilation)


# TODO: Support feature_group_count, batch_group_count and precision, and
#       allow lhs_dilation and rhs_dilation to happen at the same time.
def conv_general_dilated(lhs, rhs, window_strides, padding, lhs_dilation=None,
                         rhs_dilation=None, dimension_numbers=None,
                         feature_group_count=1, batch_group_count=1, precision=None):
  """ A general conv function that integrates normal conv, deconvolution,
  dilated convolution, etc."""
  # raise TypeError("lhs shape: {}, rhs shape: {}".format(lhs.shape, rhs.shape))
  dim = None
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  if lhs_spec != out_spec:
    raise TypeError("Current implementation requires the `data_format` of the"
                    "inputs and outputs to be the same.")
  if len(lhs_spec) >= 6:
    raise TypeError("Current implmentation does not support 4 or higher"
                    "dimensional convolution, but got: ", len(lhs_spec) - 2)
  dim = len(lhs_spec) - 2
  if lhs_dilation and rhs_dilation:
    if lhs_dilation == (1,) * dim and rhs_dilation == (1,) * dim:
      lhs_dilation, rhs_dilation = None, None
    else:
      raise TypeError("Current implementation does not support that deconvolution"
                    "and dilation to be performed at the same time, but got"
                    " lhs_dilation: {}, rhs_dilation: {}".format(lhs_dilation,
                    rhs_dilation))
  if padding not in ["SAME", "VALID"]:
    raise TypeError("Current implementation requires the padding parameter"
                    "to be either 'VALID' or 'SAME', but got: ", padding)
  # Convert params from int/Sequence[int] to list of ints.
  strides, lhs_dilation, rhs_dilation = _conv_general_param_type_converter(dim,
    window_strides, lhs_dilation, rhs_dilation
  )
  # Preprocess the shapes
  dim_maps = {}
  if isinstance(lhs_spec, str):
    dim_maps['I'] = list(rhs_spec).index('I')
    dim_maps['O'] = list(rhs_spec).index('O')
    dim_maps['N'] = list(lhs_spec).index('N')
    dim_maps['C'] = list(lhs_spec).index('C')
  else:
    dim_maps['I'] = rhs_spec[1]
    dim_maps['O'] = rhs_spec[0]
    dim_maps['N'] = lhs_spec[0]
    dim_maps['C'] = lhs_spec[1]

  spatial_dim_maps = {1: 'W', 2: "HW", 3: "DHW"}
  data_format = 'N' + spatial_dim_maps[dim] + 'C'
  tf_nn_APIs = {1: [nn.conv1d, nn.conv1d_transpose],
                2: [nn.conv2d, nn.conv2d_transpose],
                3: [nn.conv3d, nn.conv3d_transpose]}

  output = None
  if rhs_dilation or (lhs_dilation is None and rhs_dilation is None):
    output = tf_nn_APIs[dim][0](lhs, rhs, strides, padding, data_format = data_format, dilations = rhs_dilation)
  else:
    output = tf_nn_APIs[dim][1](lhs, rhs, strides, padding, data_format, lhs_dilation)

  return output
