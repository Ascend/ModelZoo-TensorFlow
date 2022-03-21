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

def data_process(data_dir, batch_size, image_feature_path, text_feature_path,output):
    '''
    :param data_dir:测试outfit json的路径
    :param batch_size:设置为16
    :return:
    '''
    # 读取数据
    # per_outfit = 8
    # G = load_graph()
    ftest = open(data_dir, 'r')  # 测试数据
    test_outfit_list = json.load(ftest)
    test_size = len(test_outfit_list)
    ############test############
    batches_fitb = int((test_size * 4) / batch_size)
    right = 0.
    for i in range(batches_fitb):
        test_fitb = load_fitb_data1(i, batch_size, test_outfit_list, image_feature_path, text_feature_path)
        # 将acc数据变为bin文件
        test_fitb[0].tofile((output+"acc_image/" + str(i) + ".bin"))
        test_fitb[1].tofile((output+"acc_text/" + str(i) + ".bin"))
        test_fitb[2].tofile((output+"acc_graph/" + str(i) + ".bin"))

    ##### AUC #####
    batches = int((test_size * 2) / batch_size)
    right = 0.
    for i in range(batches):
        test_auc = load_auc_data1(i, batch_size, test_outfit_list, image_feature_path, text_feature_path)
        # 将auc数据变为bin文件
        test_auc[0].tofile((output+"auc_image/" + str(i) + ".bin"))
        test_auc[1].tofile((output+"auc_text/" + str(i) + ".bin"))
        test_auc[2].tofile((output+"auc_graph/" + str(i) + ".bin"))



if __name__ == '__main__':

    data_dir = "./data/test_no_dup_new_100.json"
    batch_size = 16
    image_feature_path = "./data/polyvore_image_vectors/"
    text_feature_path = "./data/polyvore_text_onehot_vectors/"
    #output="/home/HwHiAiUser/AscendProjects/NGNN/data/"
    output = "./data/"
    data_process(data_dir, batch_size, image_feature_path, text_feature_path, output)
