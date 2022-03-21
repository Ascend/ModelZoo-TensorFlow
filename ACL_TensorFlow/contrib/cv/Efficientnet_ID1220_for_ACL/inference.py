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
import cv2
import json
import time


batch = 50
clear = True

def parse_label(output_path):
    label_path = output_path+"/imageLabel.npy"
    sup_labels = np.load(label_path)
    return sup_labels

def iterator_cls_inference_files(inference_path, sup_labels):
    """
    通过msame推理后的文件夹进行精度计算
    """
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    files = len(os.listdir(inference_path))
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    y_test = []
    y_pred = []
    res = []
    for file in files:
        if file.endswith(".bin"):
            tmp = np.fromfile(file, dtype='float32')
            tmpLength = len(tmp)
            tmp.resize(batch, tmpLength // batch)
            inf_label = np.argmax(tmp, axis=1)
            for i in range(batch):
                res.append(1 if inf_label[i] == sup_labels[output_num] else 0)
                output_num += 1
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num),
          "accuracy:%.6f" % (sum(res) / len(res)))
    return sum(res), len(res)

if __name__ == "__main__":
    label_path = "./data/ydata"
    inference_path = "./data/ypre/20210918_093604"
    imageLabel = parse_label(label_path)
    iterator_cls_inference_files(inference_path, imageLabel)

