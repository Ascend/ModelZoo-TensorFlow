# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import tensorflow as tf

import cifar10_data
import numpy as np
def prepro(x):
  return np.cast[np.float32]((x - 127.5) / 127.5)
DataLoader = cifar10_data.DataLoader
test_data = DataLoader('./', 'test', 12, shuffle=False)
xs=[0]
for x in test_data:
    xf = prepro(x)
    xfs = np.split(xf, 1)
    feed_dict = {xs[0]: xfs[0]}
    res=xs[0]


