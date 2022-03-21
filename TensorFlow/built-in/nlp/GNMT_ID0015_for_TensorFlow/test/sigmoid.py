# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
sigmoid
"""
import functools

import te.platform as tbe_platform
from te import tvm
from te.lang import cce as tbe
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


# pylint: disable=unused-argument,too-many-locals,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("sigmoid")
def sigmoid_compute(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    output tensor
    """
    data_input = x
    dtype = x.dtype
    exp_support = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    mul_support = tbe_platform.api_check_support("te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'x', ("float16",), dtype)

    data_input = tbe.vmins(data_input, tvm.const(11, dtype))
    data_input = tbe.vmaxs(data_input, tvm.const(-11, dtype))

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = tbe.vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = tbe.cast_to(tmp_negative, "float16")
    tmp_exp = tbe.vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = tbe.cast_to(tmp_exp, "float32")
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = tbe.broadcast(tvm.const(1, dtype), inp_shape)
        tmp_rec = tbe.vdiv(tensor_one, tmp_sum)
    else:
        tmp_rec = tbe.vrec(tmp_sum)
    return tmp_rec


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid(x, y, kernel_name="sigmoid"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        dict of x, include keys(shape and dtype)
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype")
    para_check.check_shape(shape, param_name="x")
    input_dtype = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    fused_shape = [functools.reduce(lambda a, b: a * b, shape[:])]
    data_input = tvm.placeholder(fused_shape, name="data_input", dtype=input_dtype)

    res = sigmoid_compute(data_input, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res], "disable_float_overflow": True}
    tbe.cce_build_code(sch, config)
