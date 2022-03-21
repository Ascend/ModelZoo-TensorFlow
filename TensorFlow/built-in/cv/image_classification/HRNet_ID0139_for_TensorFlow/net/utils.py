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

from collections import defaultdict
import functools
import tensorflow as tf


def create_downsample_fn_matrix(num_branches, inputs, num_output_branches, output_channels):
    '''

    :param num_branches: number of input branches
    :param num_output_branches:  number of output branches
    :return: matrix[num_branches,num_output_branches], None if not needed.
    '''

    fn_matrix = {}
    for i in range(num_branches):
        for j in range(num_output_branches):
            if j > i:
                fn_matrix[i, j] = {'input': inputs[i], 'outchannel': output_channels[j]}

    return fn_matrix


def create_upsample_fn_matrix(num_branches, inputs, num_output_branches, output_channels):
    '''

    :param num_branches: number of input branches
    :param num_output_branches:  number of output branches
    :return: matrix[num_branches][num_output_branches], None if not needed.
    '''

    fn_matrix = {}
    for i in range(num_branches):
        for j in range(num_output_branches):
            if j < i:
                fn_matrix[i, j] = {'input': inputs[i], 'outchannel': output_channels[j]}

    return fn_matrix


def add_layers(origfeatures, dwfeatrues, upfeatures, nums_output):
    '''

    :param origfeatures:
    :param dwfeatrues:
    :param upfeatures:
    :param nums_output:
    :return:
    '''

    _temp = defaultdict(list)
    for i in range(nums_output):
        for featuremaps in [origfeatures, dwfeatrues, upfeatures]:
            if i in featuremaps.keys():
                _temp[i].extend(featuremaps[i])

    outlist = []
    for i in range(nums_output):
        fmlist = _temp[i]
        add = functools.reduce(lambda a, b: a + b, fmlist)
        # for each elemwise add, go through relu
        xrelu = tf.nn.relu(add)
        outlist.append(xrelu)
    return outlist
