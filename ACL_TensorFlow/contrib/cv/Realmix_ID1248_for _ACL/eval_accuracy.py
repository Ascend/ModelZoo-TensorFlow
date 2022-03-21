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
EVAL_PATH = "/root/AscendProjects/output/20211015_214836"
if __name__ == "__main__":
    offset = 0
    output_num = 0
    right_num = 0
    files = os.listdir(EVAL_PATH)
    files.sort()
    for file in files:
        output_num += 1
        right = file.split('_')[1][0]
        with open(EVAL_PATH+'/'+file,'r') as myfile:
            eval = myfile.read()
            eval = eval.strip()
        if right == eval:
            right_num += 1
    print("准确率:",right_num/output_num*100.0)
        

