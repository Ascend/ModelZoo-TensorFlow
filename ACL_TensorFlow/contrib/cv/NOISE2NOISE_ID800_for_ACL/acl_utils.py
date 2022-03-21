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
import pickle


def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    return x.transpose([1, 2, 0])


# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return x.transpose([2, 0, 1])


def to_multichannel(i):
    if i.shape[2] == 3:
        return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def add_validation_gaussian_noise_np(x, validation_stddev=25):
    return x + np.random.normal(size=x.shape) * (validation_stddev / 255.0)


def add_validation_poisson_noise_np(x):
    chi = 30.0
    return np.random.poisson(chi * (x + 0.5)) / chi - 0.5


def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)