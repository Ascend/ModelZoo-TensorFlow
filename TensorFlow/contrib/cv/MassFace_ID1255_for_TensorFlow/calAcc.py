# coding=utf-8
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

from lib import utils
from lib import lfw
import os
import numpy as np
input_bin_dir = "/home/qyy/MassFac/dataset/lfw-bin"  #2bin生成的文件夹，需要其中的actual_issame.txt文件
#input_bin_dir = "/cache/user-job-dir/MassFac/dataset/lfw-bin/lfw-bin"

#output_bin_dir = "/cache/user-job-dir/MassFac/dataset/output-bin"
output_bin_dir =  "/home/qyy/MassFac/dataset/output-bin" #msame输出位置
paths = os.listdir(output_bin_dir)

emb_array = []
print("start!!")

for i in range(12000):
    path = str(i) +'_output_0.txt'
    if path.startswith("actual_issame"):
        continue
    else :
        with open(os.path.join(output_bin_dir, path), 'r') as file:
            content = np.array(list(map(float,file.readline().strip().split()))).astype(np.float32)
            

    emb_array.append(content)
emb_array = np.array(emb_array)
with open(os.path.join(input_bin_dir ,'actual_issame.txt'),'r') as file:
    content1 = file.readline().strip().split()
    actual_issame = [True if c=='1' else False for c in content1]
print(emb_array.shape)
emb_array = utils.l2_normalize(emb_array)
tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                     actual_issame, nrof_folds=10)

print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
print("accuracy is:")
print(accuracy)
print("over!!")