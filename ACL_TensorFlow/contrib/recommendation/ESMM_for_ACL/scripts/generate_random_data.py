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

def generate_random_data(file_path,image_num=32):
    for i in range(image_num):
        input_data = np.random.randn(1,10000,4).astype(np.float32)
        input_data.tofile(file_path+"/"+str(i.__str__().zfill(6))+".bin")
    print("num:%d random datas has been created under path:%s" %(image_num,file_path))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="./")
    parser.add_argument("--nums", type=int, default=32)
    args = parser.parse_args()

    generate_random_data(args.path,args.nums)