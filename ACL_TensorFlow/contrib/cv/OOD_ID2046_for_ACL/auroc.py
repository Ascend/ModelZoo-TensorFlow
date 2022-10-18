# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import numpy as np
import tensorflow as tf
import os

from numpy import mean
from sklearn.metrics import roc_auc_score

in_val_output_dir = "./log/result/in_val_result"
ood_val_output_dir ="./log/result/ood_val_result"

def main():
    #in-distribution validation
    in_val_output_file_list = [
        os.path.join(in_val_output_dir, x)
        for x in tf.gfile.ListDirectory(in_val_output_dir)
        if  '.txt' in x
    ]

    # ood validation
    ood_val_output_file_list = [
        os.path.join(ood_val_output_dir, x)
        for x in tf.gfile.ListDirectory(ood_val_output_dir)
        if  '.txt' in x
    ]
    i = 0
    auc_list = []
    while(i<100):
        in_loss_i = []
        ood_loss_i = []
        with open(in_val_output_file_list[i], 'r') as f:
            in_data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
            # 对list中的数据做分隔和类型转换
            for line in in_data:
                line_data = line.split()
                in_loss_i = list(map(float, line_data))
        with open(ood_val_output_file_list[i], 'r') as f:
            ood_data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
            # 对list中的数据做分隔和类型转换
            for line in ood_data:
                line_data = line.split()
                ood_loss_i = list(map(float, line_data))
                # auc using raw likelihood, larger for OOD
        neg = np.array(in_loss_i)
        pos = np.array(ood_loss_i)
        auc = roc_auc_score([0] * neg.shape[0] + [1] * pos.shape[0],
                            np.concatenate((neg, pos), axis=0))
        auc_list.append(auc)
        i = i+1

    print(mean(auc_list))



if __name__ == '__main__':
    main()
