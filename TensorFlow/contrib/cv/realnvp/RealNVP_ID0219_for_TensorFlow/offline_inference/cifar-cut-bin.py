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

import os
import numpy as np

file_prefix = "test_batch.bin"
index = 1
batch_size=12
dataset=np.fromfile(file_prefix,dtype=np.int8)
i=0
ep=0
def prepro(x):
    t=np.cast[np.float32]((x / 127.5) - 127.5)
    return t
while(i<len(dataset)):
    out=[]
    for b in range(batch_size):
        arr=np.array(dataset[i + 1:i + 3073])
        arr=np.reshape(arr,(1,3,32,32))
        a=np.transpose(arr, (0, 2, 3, 1))
        # a=np.reshape(arr,(1,32,32,3))
        b=a.flatten()
        out.extend(b)
        print(dataset[i+3073])
        i+=3073
    out=np.array(out)
    out=prepro(out)
    out.tofile("cifar_test"+str(ep)+".bin")
    ep += 1


