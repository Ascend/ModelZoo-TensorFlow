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
from scipy.io import savemat

result_dir = "./res"
ouput_blocks = [None] * 40 * 32

for i in range(1280):
    data_name = "D:/npu32/3990/20211228_21_8_12_475260/input-" + str(i) + "_output_0.txt"
    data = np.loadtxt(data_name, dtype=np.float32)
    matrix = data.reshape([256, 256])
    matrix = np.expand_dims(np.expand_dims(matrix, axis=0), axis=3)
    ouput_blocks[i] = matrix

out_mat = np.squeeze(ouput_blocks)
out_mat = out_mat.reshape([40, 32, 256, 256])

savemat(result_dir + 'ValidationCleanBlocksRaw.mat', {'results': out_mat})
