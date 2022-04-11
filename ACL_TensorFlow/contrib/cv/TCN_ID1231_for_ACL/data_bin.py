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
# Imports
import numpy as np
from utils import data_generator
T=1000
seq_len = 10
n_test = 1
f = open("test.txt" , 'w')
for i in range(5):
    test_x, test_y = data_generator(T, seq_len, n_test)
    test_x = np.reshape(test_x, test_x.shape+ (1,))
    print(test_x)
    print(test_x.shape)
    f.write("test_x"+ str(i+1) + str(test_x) + '\n' + "test_y"+ str(i+1) + str(test_y) + '\n')
    test_x.tofile("test"+ str(i+1) + ".bin") # 处理后的图片保存为bin文件
    test = np.fromfile("test"+ str(i+1) + ".bin")
    print(test.shape)
    print(test)

