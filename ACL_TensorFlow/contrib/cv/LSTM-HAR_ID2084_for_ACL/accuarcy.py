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

import sys
import os
import numpy as np
import tensorflow as tf

def one_hot(y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS

def read_label(label_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
    file = open(label_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return (y_ - 1)


if __name__ == "__main__":
    output_path = sys.argv[1]
    label_path = sys.argv[2]
    label_dict = read_label(label_path)
    output_num = 0
    check_num = 0
    files = os.listdir(output_path)
    files.sort(key=lambda x:int(str(x).split('_')[0]))
    print(label_dict)
    for file in files:
        if file.endswith(".bin"):
            tmp = np.fromfile(output_path+'/'+file, dtype=np.int32)
            # print(tmp.shape)
            print("------------------")
            print(tmp)
            print(label_dict[output_num])
            if(tmp == label_dict[output_num]):
                check_num += 1
            output_num += 1
            
            # print(label_dict.shape)
    #         try:
    #             pic_name = str(file.split(".JPEG")[0])+".JPEG"
    #             print("%s, inference label:%d, gt_label:%d"%(pic_name,inf_label,label_dict[pic_name]))
    #             if inf_label == label_dict[pic_name]:
    #                 #print("%s inference result Ok!"%pic_name)
    #                 check_num += 1
    #         except:
    #             print("Can't find %s in the label file: %s"%(pic_name,label_path))
    top1_accuarcy = check_num/output_num
    print("Totol test dataset num: %d, Top1 accuarcy: %.4f"%(output_num,top1_accuarcy))

