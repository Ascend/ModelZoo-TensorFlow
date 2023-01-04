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

import argparse
import numpy as np
import os

def generate_random_data(file_path,image_num=32,batchsize=1):
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    sub_dir1 = os.path.join(file_path,'Input-Segment')
    sub_dir2 = os.path.join(file_path,'Input-Token')
    if not os.path.exists(sub_dir1):
            os.makedirs(sub_dir1)
    if not os.path.exists(sub_dir2):
            os.makedirs(sub_dir2)
    for i in range(image_num):
        input_data1 = np.random.rand(batchsize,1024).astype(np.float32)
        input_data2 = np.random.rand(batchsize,1024).astype(np.float32)
        input_data1.tofile(os.path.join(sub_dir1,str(i.__str__().zfill(6))+".bin"))
        input_data2.tofile(os.path.join(sub_dir2,str(i.__str__().zfill(6))+".bin"))
    print("num:%d random datas has been created under path:%s" %(image_num,file_path))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="./")
    parser.add_argument("--nums", type=int, default=32)
    parser.add_argument("--batchsize", type=int, default=1)
    args = parser.parse_args()

    generate_random_data(args.path,args.nums,args.batchsize)
