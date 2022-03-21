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
from npu_bridge.npu_init import *
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import utils
from model import imm


def XycPackage():
    """
    Load Dataset and set package.
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    noOfTask = 3
    x = []
    x_ = []
    y = []
    y_ = []
    xyc_info = []

    x.append(np.concatenate((mnist.train.images,mnist.validation.images)))
    y.append(np.concatenate((mnist.train.labels,mnist.validation.labels)))
    x_.append(mnist.test.images)
    y_.append(mnist.test.labels)
    xyc_info.append([x[0], y[0], 'train-idx1'])

    for i in range(1, noOfTask):
        idx = np.arange(784)                 # indices of shuffling image
        np.random.shuffle(idx)
        
        x.append(x[0].copy())
        x_.append(x_[0].copy())
        y.append(y[0].copy())
        y_.append(y_[0].copy())

        x[i] = x[i][:,idx]           # applying to shuffle
        x_[i] = x_[i][:,idx]

        xyc_info.append([x[i], y[i], 'train-idx%d' % (i+1)])

    for i in range(noOfTask):
        xyc_info.append([x_[i], y_[i], 'test-idx%d' % (i+1)])

    return x, y, x_, y_, xyc_info

