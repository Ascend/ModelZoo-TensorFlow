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
import os
from operator import truediv

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score



def AA_andEachClassAccuracy(confusion_matrix):
    # 根据混淆矩阵计算准确率及其平均准确率，返回准确率和平均准确率
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

predict_file = "D:\PytorchProgram\HY\\2022223_10_28_5_968623"
label_file = "D:\PytorchProgram\HY\IPytest"

def read_file(predict_file, label_file):
    files_predict = os.listdir(predict_file)
    # print(files_predict)
    files_label = os.listdir(label_file)

    pred = list()
    label = list()
    for files in files_predict:
        if files.endswith(".txt"):
            # print(files)
            tmp = np.loadtxt(predict_file + '/' + files, dtype='float32')
            pred.append(tmp)

    for file in files_label:
        if file.endswith('.bin'):
            tmp = np.fromfile(label_file + '/' + file, dtype='float32')
            label.append(tmp)
    return pred, label

pred, label = read_file(predict_file, label_file)
confusion = confusion_matrix(np.argmax(label, axis=1), np.argmax(pred, axis=1))
each_acc, aa = AA_andEachClassAccuracy(confusion)
oa = accuracy_score(np.argmax(label, axis=1), np.argmax(pred, axis=1))
kappa = cohen_kappa_score(np.argmax(label, axis=1), np.argmax(pred, axis=1))


print(aa*100)
print(oa*100)
print(kappa*100)
