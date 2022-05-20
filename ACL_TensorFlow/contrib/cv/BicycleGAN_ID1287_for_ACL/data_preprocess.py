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


import os
import numpy as np
import argparse
import glob
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--Z_dim", type=int, default=8,help='size of the latent vector')
parser.add_argument("--sample_num", type=int, default=20,help='number of latent vectors sampled')
parser.add_argument("--data_num", type=int, default=1098,help='number of input data,one input has sample_num latent vectors')
parser.add_argument("--output_path", type=str, default=os.path.dirname(os.path.realpath(__file__)),help='path of bin files to save')
parser.add_argument("--data_path", type=str, default='',help='path of datasets')

args = parser.parse_args()


def load_images(path, image_size):
    test_all = sorted(glob.glob(os.path.join(path, "val/*.jpg")))

    test_input = []
    for img in test_all:
        full_image = Image.open(img)
        full_image = np.asarray(full_image.resize((2 * image_size, image_size), Image.BICUBIC))

        test_input.append(full_image[:, full_image.shape[1] // 2:, :] / 255.)

    # need to normalize to [-1,1]
    return np.asarray(test_input) * 2 - 1

def make_bin(input_path, image_size,output_path):
    # making latent vectors bin files,every input image need args.sample_num
    # latent vectors
    for i in range(args.sample_num):
        path = os.path.join(output_path,"z",f'z_{i}')
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(args.data_num):
            #print(args.Z_dim)
            z = np.random.normal(size=(1,args.Z_dim)).astype(np.float32)
            print(z.shape)
            z.tofile(os.path.join(path,f'{j+1}.bin'))

    # making input images bin files
    test_A = load_images(input_path,image_size)
    path = os.path.join(output_path,"input")
    if not os.path.exists(path):
        os.makedirs(path)
    for index,item in enumerate(test_A):
        bin = np.expand_dims(item,axis=0).astype(np.float32)
        bin.tofile(os.path.join(path,f'{index+1}.bin'))

make_bin(args.data_path,256,args.output_path)
