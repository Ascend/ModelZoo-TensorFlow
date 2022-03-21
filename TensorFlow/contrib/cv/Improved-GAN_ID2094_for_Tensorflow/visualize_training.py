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
from npu_bridge.npu_init import *
import h5py
import numpy as np
import imageio
import glob, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--h', type=int, default=32)
parser.add_argument('--w', type=int, default=32)
parser.add_argument('--c', type=int, default=3)
parser.add_argument('--n', type=int, default=8)
args = parser.parse_args()

if not args.train_dir or not args.output_file:
    raise ValueError("Please specify train_dir and output_file")

II = []
for file in sorted(glob.glob(os.path.join(args.train_dir, "*.hdf5")), key=os.path.getmtime):
    print (file)
    f = h5py.File(file, 'r')
    I = np.zeros((args.n*args.h, args.n*args.w, args.c))
    for i in range(args.n):
        for j in range(args.n):
            I[args.h*i:args.h*(i+1), args.w*j:args.w*(j+1), :] = f[f.keys()[0]][i*args.n+j,:,:,:]
    II.append(I)

II = np.stack(II)
print (II.shape)
imageio.mimsave(args.output_file, II, fps=5)

