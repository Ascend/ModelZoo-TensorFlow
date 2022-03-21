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
from npu_bridge.npu_init import *
import scipy.io as sio
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def getData(filename):
    load_data = sio.loadmat(filename)
    y = load_data['y']
    X = load_data['X'].transpose(3, 0, 1, 2)
    return X, y


if __name__ == '__main__':
    file = './data/train_32x32.mat'
    X, y = getData(filename=file)
    print(X.shape)