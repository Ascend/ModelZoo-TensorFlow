#
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
#
# Authors : Florian Lalande < florianlalande@orange.fr >
#           Austin Peel < austin.peel@cea.fr >
import tensorflow as tf
import json
import numpy as np
#from load_data_multimodal import load_graph, load_fitb_data1, load_auc_data1
from datetime import *
def data_process(data_dir,output):
    '''
    :param data_dir:测试outfit json的路径
    :param batch_size:设置为16
    :return:
    '''
    # 读取数据
    # per_outfit = 8
    # G = load_graph()
    #ftest = open(data_dir, 'r')  # 测试数据
    data1 = np.load(data_dir)
    X = data1
    X = X.reshape(1024, 1, 4, 5, 100)
    numbers=np.arange(10)
    #X[1000].tofile(output+"data1000"+".bin")
    X_test = X[numbers]
    for i in range(10):
        X_test[i].tofile(output+"data"+str(i)+".bin")




if __name__ == '__main__':

    data_dir = "./mgcnn-master/data/sigma035/pdf_j37.npy"
    batch_size=16
    #image_feature_path="F:/NGNN/data/polyvore_image_vectors/"
    #text_feature_path="F:/NGNN/data/polyvore_text_onehot_vectors/"
    #output="/home/HwHiAiUser/AscendProjects/NGNN/data/"
    output="./mgcnn-master/data/sigma035/"
    data_process(data_dir, output)