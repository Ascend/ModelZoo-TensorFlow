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
# =============================================================================
import os
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result", type=str, default="./output")
    parser.add_argument("--ground_truth", type=str, default="./ground_truth")
    parser.add_argument("--output_index", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dtype", type=str, default='float32')   #datatype of bin files
    args = parser.parse_args()

    image_cnt = 0
    acc_cnt = 0
    infer_list = os.listdir(args.infer_result)
    infer_list.sort()
    for file in infer_list:
        print("start to process {}".format(file))
        index = file.split("davinci_")[1].split("_output")[0]
        infer_res = np.fromfile(os.path.join(args.infer_result,file), dtype=args.dtype).astype('int32')
        gt_res = np.fromfile(os.path.join(args.ground_truth,"{}.bin".format(index)),dtype='int32')
        image_cnt += infer_res.shape[0]
        acc_cnt += np.sum(infer_res == gt_res)
    print("Accuarcy: %.3f"%(acc_cnt/image_cnt))
