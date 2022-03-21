"""License"""
from npu_bridge.npu_init import *
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
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

f = h5py.File('Dataset2.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
a = f['data'][:]
sources = np.transpose(a, (0, 3, 2, 1))

vis = sources[100, :, :, 0]
ir = sources[100, :, :, 1]

ir_ds = scipy.ndimage.zoom(ir, 0.25)
ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)

fig = plt.figure()
V = fig.add_subplot(221)
I = fig.add_subplot(222)
I_ds = fig.add_subplot(223)
I_ds_us = fig.add_subplot(224)

V.imshow(vis, cmap = 'gray')
I.imshow(ir, cmap = 'gray')
I_ds.imshow(ir_ds, cmap = 'gray')
I_ds_us.imshow(ir_ds_us, cmap = 'gray')
plt.show()
# print
# 'Resampled by a factor of 2 with nearest interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 0)
#
# print
# 'Resampled by a factor of 2 with bilinear interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 1)
#
# print
# 'Resampled by a factor of 2 with cubic interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 3)
#
# print
# 'Downsampled by a factor of 0.5 with default interpolation:'
# print(scipy.ndimage.zoom(x, 0.5))

