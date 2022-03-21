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
# ============================================================================
import tensorflow as tf
import json
import numpy as np
from load_data_multimodal import load_graph, load_fitb_data1, load_auc_data1
from datetime import *
import os

def get_acc(data_dir, batch_size, predicted_score_dir):
    '''
    :param pb_path:pb文件的路径
    :param data_dir:测试outfit json的路径
    :param batch_size:设置为16
    :return:
    '''
    ftest = open(data_dir, 'r')  # 测试数据
    test_outfit_list = json.load(ftest)
    test_size = len(test_outfit_list)
    ############test############
    batches = int((test_size * 2) / batch_size)
    print(test_size)
    print(batches)

    right=0.
    count=0
    files = os.listdir(predicted_score_dir)
    for file in files:
        position=predicted_score_dir+file
        print(position)
        with open(position, "r", encoding='utf-8') as f:  # 打开文件
            for j in range(int(batch_size / 2)):
                a = []
                for k in range(j * 2, (j + 1) * 2):
                    line=f.readline().splitlines()
                    line=list(map(float,line))
                    #line=map(float,line)
                    #float(line)
                    a.append(line)
                    #a.append(f.readline())
                print(str(j)+" "+str(k))
                print(a)
                if np.argmax(a) == 0:
                    right += 1.
                print(right)
        f.close()
        count+=1
        if count%10==0:
            print(str(count)+"files has been computed!")

    #print("over")
    auc = float(right / test_size)

    print('now():' + str(datetime.now()))
    print("Test:", "auc:", "{:.9f}".format(auc))



if __name__ == '__main__':

    data_dir = "./data/test_no_dup_new_100.json"
    batch_size=16
    #predicted_score_dir="./output/predicted_acc_1/"
    predicted_score_dir = "./predicted_auc_1/"

    get_acc(data_dir, batch_size, predicted_score_dir)
