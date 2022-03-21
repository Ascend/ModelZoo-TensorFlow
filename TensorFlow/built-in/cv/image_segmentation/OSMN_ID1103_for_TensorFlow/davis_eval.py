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
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

data_path = sys.argv[1]
pred_path = sys.argv[2] #'DAVIS/Results/Segmentations/480p/OSVOS'
dataset_version = sys.argv[3]
dataset_split = sys.argv[4]
if len(sys.argv) > 5:
    vis_path = sys.argv[5]
else:
    vis_path = None
listFile = '%s/ImageSets/%s/%s.txt' % (data_path, dataset_version, dataset_split)
gt_path = os.path.join(data_path, 'Annotations', '480p')
with open(listFile, 'r') as f:
    fds = [line.strip() for line in f]
im_num = 0
iou =[]
seq_n = 0
sample_n = 0
subfd_names = []
for i, fd in enumerate(fds):
    print(fd)
    file_list = os.listdir(os.path.join(gt_path, fd))
    im_list = [name for name in file_list if len(name) > 4 and name[-4:]=='.png']
    im_list = sorted(im_list)
    im_list = im_list[1:-1] # remove first and last image
    pred_list = os.listdir(os.path.join(pred_path, fd))
    if dataset_version == '2017':
        sub_fds = [name for name in pred_list if len(name) < 4]
        sub_fds = sorted(sub_fds)
        print(sub_fds)
        for sub_fd in sub_fds:
            subfd_names.append(fd+'/'+sub_fd)
    iou_seq = []
    for i,im_name in enumerate(im_list):
        iou_im = 0
        scores = []
        label_gt = np.array(Image.open(os.path.join(gt_path, fd, im_name)))
        if dataset_version == '2017':
            for j, sub_fd in enumerate(sub_fds):

                score = np.load(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + '.npy'))
                scores.append(score)
            im_size = scores[0].shape
            bg_score = np.ones(im_size) * 0.5
            scores = [bg_score] + scores
            score_all = np.stack(tuple(scores), axis = -1)
            class_n = score_all.shape[2] - 1
            label_pred = score_all.argmax(axis=2)
        else:
            class_n = 1
            label_gt = label_gt > 0
            label_pred = np.array(Image.open(os.path.join(pred_path,fd, im_name))) > 0 
        label_pred = np.array(Image.fromarray(label_pred.astype(np.uint8)).resize((label_gt.shape[1], label_gt.shape[0]), 
            Image.NEAREST))
        #cv2.resize(label_pred, label_gt.shape, label_pred, 0, 0, cv2.INTER_NEAREST)
        if vis_path:
            res_im = Image.fromarray(label_pred, mode="P")
            res_im.putpalette(PALETTE)
            if not os.path.exists(os.path.join(vis_path, fd)):
                os.makedirs(os.path.join(vis_path, fd))
            res_im.save(os.path.join(vis_path, fd, im_name))
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


