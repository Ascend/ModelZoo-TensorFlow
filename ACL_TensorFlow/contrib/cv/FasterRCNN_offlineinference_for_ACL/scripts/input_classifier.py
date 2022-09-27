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
import numpy as np
from tqdm import tqdm
from PIL import Image
from anchors import get_anchors
from bbox import BBoxUtility

image_ids = open("/root/FasterRCNN/VOCdevkit/VOC2012/ImageSets/Main/val.txt").read().strip().split()

def cvtColor(img):
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
        return img
    else:
        img = img.convert('RGB')
        return img

bbox_util = BBoxUtility(21, nms_iou=0.5, min_k=150)

for image_id in tqdm(image_ids):
    image_path = '/root/FasterRCNN/VOCdevkit/VOC2012/JPEGImages/' + image_id + '.jpg'
    image = Image.open(image_path)
    image = cvtColor(image)
    image = np.array(image, dtype='float32')
    image = image[np.newaxis, :]
    if image.shape[1] == 800:
        rpn_pred0 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x800/davinci_' + image_id + '_output0.bin', dtype='float32')
        rpn_pred0.shape = 1, 17100, 1
        rpn_pred1 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x800/davinci_' + image_id + '_output1.bin', dtype='float32')
        rpn_pred1.shape = 1, 17100, 4
        rpn_pred2 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x800/davinci_' + image_id + '_output2.bin', dtype='float32')
        rpn_pred2.shape = 1, 50, 38, 1024

        rpn_pred = [rpn_pred0, rpn_pred1, rpn_pred2]

        anchors = get_anchors([800, 600], 'resnet50', [128, 256, 512])

        rpn_results = bbox_util.detection_out_rpn(rpn_pred, anchors)

        rpn_pred2.tofile('/root/FasterRCNN/FasterRCNN-PR/input_classifier_x800_0/' + image_id + '.bin')
        rpn_results[:, :, [1, 0, 3, 2]].tofile('/root/FasterRCNN/FasterRCNN-PR/input_classifier_x800_1/' + image_id + '.bin')

    else:
        rpn_pred0 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x600/davinci_' + image_id + '_output0.bin', dtype='float32')
        rpn_pred0.shape = 1, 17100, 1
        rpn_pred1 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x600/davinci_' + image_id + '_output1.bin', dtype='float32')
        rpn_pred1.shape = 1, 17100, 4
        rpn_pred2 = np.fromfile('/root/FasterRCNN/FasterRCNN-PR/results_rpn_x600/davinci_' + image_id + '_output2.bin', dtype='float32')
        rpn_pred2.shape = 1, 38, 50, 1024

        rpn_pred = [rpn_pred0, rpn_pred1, rpn_pred2]

        anchors = get_anchors([600, 800], 'resnet50', [128, 256, 512])

        rpn_results = bbox_util.detection_out_rpn(rpn_pred, anchors)

        rpn_pred2.tofile('/root/FasterRCNN/FasterRCNN-PR/input_classifier_x600_0/' + image_id + '.bin')
        rpn_results[:, :, [1, 0, 3, 2]].tofile('/root/FasterRCNN/FasterRCNN-PR/input_classifier_x600_1/' + image_id + '.bin')
