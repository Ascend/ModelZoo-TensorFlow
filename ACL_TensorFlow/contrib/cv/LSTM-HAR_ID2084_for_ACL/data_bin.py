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
import tensorflow as tf
import os
def load_X(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

data_dir = "./input"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

X_test_signals_paths = [
        "./UCI HAR Dataset/test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test = load_X(X_test_signals_paths)
print(X_test.shape[0])
print(X_test[0].shape)
for i in range(X_test.shape[0]):
    temp = X_test[i].reshape((1, 128, 9))
    # print(temp.shape)
    temp.tofile("./input/" + str(i) + ".bin")
print("Done!")
