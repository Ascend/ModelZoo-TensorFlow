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
# ==============================================================================

import cv2
import numpy as np

import os
import sys

bbox_data_dir = 'labels/train2014'
img_dir = 'train2014'
res_file = os.open('train_annotation.txt', 'w')

bbox_data_files = os.listdir(bbox_data_dir)

for bbox_file in bbox_data_files:
    img_name = bbox_file.split('.')[0]+'.jpg'
    res_file.write(img_name)
    with open(bbox_data_dir+'/'+bbox_file, 'r') as f:
        bboxes = f.readlines()
    for line in bboxes:
        bbox = line.strip().split()
        class_idx = int(bbox[0])
        x_center = float(bbox[1])
        y_center = float(bbox[2])
        w_bbox = float(bbox[3])
        h_bbox = float(bbox[4])
        xmin = x_center - w_bbox/2.
        ymin = y_center - h_bbox/2.
        xmax = x_center + w_bbox/2.
        ymax = y_center + h_bbox/2.
        res_file.write(' '+str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)+','+str(class_idx))
    res_file.write('\n')

