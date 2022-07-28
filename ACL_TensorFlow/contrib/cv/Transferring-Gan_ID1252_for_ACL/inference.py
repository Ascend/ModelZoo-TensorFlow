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

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import tflib as lib

inception_path = "./inception"
import tflib.inception_score

fpath = "/root/transfergan/transfergan/inference/output4/2022728_10_2_17_375490/{}_output_0.bin"

def get_inception_score(n):
    all_samples = []
    for i in range(n // 100):
        temp = np.fromfile(fpath.format(i),dtype=np.float32)
        all_samples.append(temp.reshape(100,-1))
    # return 
    # print("test")
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

print(get_inception_score(50000))