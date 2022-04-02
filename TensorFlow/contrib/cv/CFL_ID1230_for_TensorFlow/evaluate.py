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

import numpy as np
from numpy import *
import os
from config import *
import glob
from PIL import Image
import tensorflow as tf
import imageio



def evaluate():

    prediction_path_list=[]
    prediction_path_list.append("p1_EM_test.jpg")
    prediction_path_list.append("p1_CM_test.jpg")
    gt_path_list=[]
    gt_path_list.append("p1_EM.jpg")
    gt_path_list.append("p1_CM.jpg")
    P, R, Acc, f1, IoU = [], [], [], [], []
    prediction = Image.open(prediction_path_list[0])
    for im in range(len(prediction_path_list)):
        # predicted image
        prediction = Image.open(prediction_path_list[im])
        pred_W, pred_H = prediction.size
        prediction = np.array(prediction) / 255.
        # gt image
        gt = Image.open(gt_path_list[im])
        gt = gt.resize([pred_W, pred_H])
        gt = np.array(gt) / 255.
        gt = (gt >= 0.01).astype(int)

        th = 0.1
        tp = np.sum(np.logical_and(gt == 1, prediction > th))
        tn = np.sum(np.logical_and(gt == 0, prediction <= th))
        fp = np.sum(np.logical_and(gt == 0, prediction > th))
        fn = np.sum(np.logical_and(gt == 1, prediction <= th))

        # How accurate the positive predictions are
        P.append(tp / (tp + fp))
        # Coverage of actual positive sample
        R.append(tp / (tp + fn))
        # Overall performance of model
        Acc.append((tp + tn) / (tp + tn + fp + fn))
        # Hybrid metric useful for unbalanced classes
        f1.append(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        # Intersection over Union
        IoU.append(tp / (tp + fp + fn))

    return np.mean(P), np.mean(R), np.mean(Acc), np.mean(f1), np.mean(IoU)

if __name__ == '__main__':
    P,R,Acc,f1,IoU=evaluate()
    print("P:",P)
    print("R:",R)
    print("Acc:",Acc)
    print("f1:",f1)
    print("IoU:",IoU)