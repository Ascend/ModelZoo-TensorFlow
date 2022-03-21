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


import json
import numpy as np
import cv2
from pycocotools import mask as COCOmask

def showMask(img_obj):
    img = cv2.imread(img_obj['fpath'])
    img_ori = img.copy()
    gtmasks = img_obj['gtmasks']
    n = len(gtmasks)
    print(img.shape)
    for i, mobj in enumerate(gtmasks):
        if not (type(mobj['mask']) is list):
            print("Pass a RLE mask")
            continue
        else:
            pts = np.round(np.asarray(mobj['mask'][0]))
            pts = pts.reshape(pts.shape[0] // 2, 2)
            pts = np.int32(pts)
            color = np.uint8(np.random.rand(3) * 255).tolist()
            cv2.fillPoly(img, [pts], color)
    cv2.addWeighted(img, 0.5, img_ori, 0.5, 0, img)
    cv2.imshow("Mask", img)
    cv2.waitKey(0)

def get_seg(height, width, seg_ann):
    label = np.zeros((height, width, 1))
    if type(seg_ann) == list or type(seg_ann) == np.ndarray:
        for s in seg_ann:
            poly = np.array(s, np.int).reshape(len(s)//2, 2)
            cv2.fillPoly(label, [poly], 1)
    else:
        if type(seg_ann['counts']) == list:
            rle = COCOmask.frPyObjects([seg_ann], label.shape[0], label.shape[1])
        else:
            rle = [seg_ann]
        # we set the ground truth as one-hot
        m = COCOmask.decode(rle) * 1
        label[label == 0] = m[label == 0]
    return label[:, :, 0]
