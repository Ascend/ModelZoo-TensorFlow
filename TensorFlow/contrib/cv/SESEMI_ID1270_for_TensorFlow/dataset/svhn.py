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
"""
Download SVHN cropped digit classification dataset from:
http://ufldl.stanford.edu/housenumbers/
"""

import numpy as np
import scipy.io
import os


def load_data():
    """Loads SVHN dataset.
    
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = './datasets/svhn'
    
    fpath = os.path.join(dirname, 'train_32x32.mat')
    d = scipy.io.loadmat(fpath)
    x_train = np.transpose(d['X'], (3, 0, 1, 2))
    y_train = d['y'].reshape(-1) # shape=(len(y),)
    y_train[y_train == 10] = 0 # re-assign label 0 to digit zero

    fpath = os.path.join(dirname, 'test_32x32.mat')
    d = scipy.io.loadmat(fpath)
    x_test = np.transpose(d['X'], (3, 0, 1, 2))
    y_test = d['y'].reshape(-1) # shape=(len(y),)
    y_test[y_test == 10] = 0 # re-assign label 0 to digit zero

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    return (x_train, y_train), (x_test, y_test)

