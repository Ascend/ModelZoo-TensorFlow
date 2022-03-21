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

from scipy.io import loadmat
import numpy as np
import os

val_dir = './data/ValidationNoisyBlocksRaw.mat'
mat = loadmat(val_dir)
val_img = mat['ValidationNoisyBlocksRaw']  # (40, 32, 256, 256)
val_img = val_img.reshape([1280, 256, 256])

for i in range(len(val_img)):
    each_block = val_img[i]  # (256, 256)
    each_block = np.expand_dims(np.expand_dims(each_block, axis=0), axis=3)
    data_name = "./data/input-" + str(i) + ".bin"
    if os.path.exists(data_name):
        os.remove(data_name)
    with open(data_name, 'wb') as fp:
        fp.write(np.ascontiguousarray(each_block))
    fp.close()
