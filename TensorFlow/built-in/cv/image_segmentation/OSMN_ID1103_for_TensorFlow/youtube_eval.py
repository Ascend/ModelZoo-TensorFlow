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
from npu_bridge.npu_init import *
import os
import sys
import cv2
from PIL import Image
import numpy as np
from util import calcIoU
dataset_path = sys.argv[1] # '/raid/ljyang/data/youtube_masks/Labels'
pred_path = sys.argv[2] #'DAVIS/Results/Segmentations/480p/OSVOS'
listFile=os.path.join(dataset_path, 'all.txt')
gt_path = os.path.join(dataset_path, 'Labels')
with open(listFile, 'r') as f:
    fds = [line.strip() for line in f]
im_num = 0
iou =[]
seq_n = 0
sample_n = 0
subfd_names = []
class_n = 1
for i, fd in enumerate(fds):
    print(fd)
    im_list = [name for name in os.listdir(os.path.join(gt_path,fd)) ] 
    im_list = sorted(im_list)
    if len(im_list) < 2:
        continue
    pred_list = os.listdir(os.path.join(pred_path, fd))
    iou_seq = []
    for im_name in im_list[1:]:
        iou_im = 0
        scores = []
        label_gt = np.array(Image.open(os.path.join(gt_path, fd, im_name))) > 0
        label_pred = np.array(Image.open(os.path.join(pred_path, fd, im_name))) > 0
        label_pred = np.array(Image.fromarray(label_pred.astype(np.uint8)).resize((label_gt.shape[1], label_gt.shape[0]), 
            Image.NEAREST))
        #cv2.resize(label_pred, label_gt.shape, label_pred, 0, 0, cv2.INTER_NEAREST)
        iou_seq.append(calcIoU(label_gt, label_pred, class_n))
    iou_seq = np.stack(iou_seq, axis=1)
    print(iou_seq.mean(axis=1))
    sample_n += iou_seq.size
    iou.extend(iou_seq.mean(axis=1).tolist())#flatten and append
iou = np.array(iou)
print("iou:", iou.mean())
with open("iou.txt", "w") as f:
    for fd, num in zip(subfd_names, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())


