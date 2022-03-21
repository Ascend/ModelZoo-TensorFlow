"""
om precision
"""
# coding=utf-8
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

from models.pre_input import r2c, myPSNR, nMse
import numpy as np

psnrRec, mseRec = [], []
for i in range(36):
    label = np.fromfile('test_bin/label/{}.bin'.format(i), dtype='float32').reshape([256, 256, 2])
    pred = np.fromfile('test_bin/2021122_17_57_36_36401/{}_output_0.bin'.format(i), dtype='float32').reshape(
        [256, 256, 2])
    normOrg = np.abs(r2c(label))
    normRec = np.abs(r2c(pred))
    psnrRec.append(myPSNR(normOrg, normRec))
    mseRec.append(nMse(normOrg, normRec))

print(psnrRec)
print(mseRec)
print(np.mean(psnrRec))
print(np.mean(mseRec))
# [35.78039691579907, 38.43462532439622, 39.17218824181286, 40.87208849357368, 36.672110548153306, 40.317175897294106, 39.40752125339972, 36.3313990955383, 39.47057475822636, 35.70946107177585, 38.40860076220202, 40.458443998876646, 35.66135353710752, 45.003707663478004, 40.38988686547921, 36.112803291327936, 42.41935582440858, 40.36662264535356, 33.270888243705, 35.52936821292876, 47.57913027967082, 44.22590419068898, 43.97678488973056, 37.671656114711894, 41.689774357852755, 44.502963211020415, 39.93984505580406, 41.718674466424794, 38.68058080410507, 33.88661907222464, 39.67466298587127, 38.804814693211625, 40.76882495422525, 37.93415810448457, 41.41929811747059, 43.404795996618176]
# [0.0048713093, 0.0028090903, 0.0035490794, 0.0026591762, 0.0034545711, 0.0024011524, 0.0026533394, 0.00407295, 0.0022362445, 0.00502619, 0.0029184497, 0.0026124148, 0.0044474257, 0.0022724718, 0.0042174105, 0.0039152955, 0.0025808353, 0.0028095306, 0.004384273, 0.0031981145, 0.0017023439, 0.002365951, 0.003092263, 0.0029181961, 0.002148601, 0.0027559346, 0.0027325805, 0.0012116899, 0.001961604, 0.004019271, 0.0031205215, 0.003133437, 0.002353341, 0.003973036, 0.0036211782, 0.0025614544]
# 39.601862776082
# 0.0030766868
