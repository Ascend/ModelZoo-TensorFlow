# Copyright 2022 Huawei Technologies Co., Ltd
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

import tensorflow as tf

__all__ = ['split']


def split(x, num_or_size_splits, axis=0, keep_dims=False):
    """Split the tensor with possibly reduced dimension.

    Args:
        x: tensor, the source tensor to split.
        num_or_size_splits: int or list[int]. If is given `int`, specifying
            the number of the splits; if given list[int], then the summation
            of the sizes should equal to the length of the `axis` of x.
        axis: int, which axis to split.
        keep_dims: boolean, whether to reduce the `axis` dimension after split.
            Dafault to False.

    Returns:
        list[tensor]
    """
    x_list = tf.split(x, num_or_size_splits, axis)

    if not keep_dims:
        x_list2 = [tf.squeeze(x_, axis) for x_ in x_list]
        return x_list2

    return x_list

