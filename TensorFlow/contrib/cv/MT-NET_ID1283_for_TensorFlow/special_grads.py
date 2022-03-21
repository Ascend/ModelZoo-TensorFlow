#
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
#
""" Code for second derivatives not implemented in TensorFlow library. """
from npu_bridge.npu_init import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
    gradient = gen_nn_ops._max_pool_grad(op.inputs[0], op.outputs[0],
            grad, op.get_attr("ksize"), op.get_attr("strides"),
            padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
    gradgrad1 = array_ops.zeros(shape = array_ops.shape(op.inputs[1]), dtype=gradient.dtype)
    gradgrad2 = array_ops.zeros(shape = array_ops.shape(op.inputs[2]), dtype=gradient.dtype)
    return (gradient, gradgrad1, gradgrad2)

