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
 
 
def data_generator(n, seq_length, seed):
    """
    Args:
        seq_length: Length of the adding problem data
        n: # of data in the set
    """
    np.random.seed(seed)
    x_num = np.random.uniform(0, 1, (n, 1, seq_length))
    x_mask = np.zeros([n, 1, seq_length])
    y = np.zeros([n, 1])
    for i in range(n):
        l = seed*i
        np.random.seed(l)
        positions = np.random.choice(seq_length, size=2, replace=False)
        x_mask[i, 0, positions[0]] = 1
        x_mask[i, 0, positions[1]] = 1
        y[i, 0] = x_num[i, 0, positions[0]] + x_num[i, 0, positions[1]]
    x = np.concatenate((x_num, x_mask), axis=1)
    x = np.transpose(x, (0, 2, 1))
    return x, y
 
 
if __name__ == '__main__':
    for i in range(50):
        x_train, y_train = data_generator(n=1, seq_length=600, seed=i)
        x_train = np.asarray(x_train).astype('float32')
        j = i + 10
        x_train.tofile("C:/Users/Administrator/Desktop/mytest/my_test_new" + "/" + "test" + str(j) + ".bin")