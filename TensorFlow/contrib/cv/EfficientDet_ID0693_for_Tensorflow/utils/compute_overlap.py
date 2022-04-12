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
import numpy as np

def compute_overlap(a, b):
    #a [N,4]
    #b [M,4]
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0]) + 1
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) + 1
    # 假设a的数目是N，b的数目是M
    # np.expand_dims((N,),axis=1)将(N,)变成(N,1)
    # np.minimum((N,1),(M,)) 得到 (N M) 的矩阵 代表a和b逐一比较的结果
    # 取x和y中较小的值 来计算intersection
    # iw和ih分别是intersection的宽和高 iw和ih的shape都是(N,M), 代表每个anchor和groundTruth之间的intersection
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0) #不允许iw或者ih小于0

    ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) *(a[:, 3] - a[:, 1] + 1), axis=1) + area - iw * ih
    # 并集的计算 S_a+S_b-interection_ab
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih
    return intersection / ua # (N,M)