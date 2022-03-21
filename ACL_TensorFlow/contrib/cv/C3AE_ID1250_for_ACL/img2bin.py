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

import cv2
import mxnet as mx
import numpy as np
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
import os

MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)

def gen_boundbox(box, landmark):
    # gen trible boundbox
    ymin, xmin, ymax, xmax = map(int, [box[1], box[0], box[3], box[2]])
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark[2], landmark[2+5])
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # 包含五官最小的框
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])

def gen_face(detector, image, image_path="", only_one=True):
    ret = detector.detect_face(image) 
    if not ret:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = ret
    if only_one and len(bounds) > 1:
        raise Exception("more than one face %s"%image_path)
    return ret

def predict(img, file_name):
    try:
        bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
        ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)
    except Exception as ee:
        ret = None
        print(ee)
    if not ret:
        print("no face")
        return img, None
    padding = 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    bounds, lmarks = bounds, lmarks

    colors = [(0, 0, 255), (0, 0, 0), (255, 0, 0)]
    for pidx, (box, landmarks) in enumerate(zip(bounds, lmarks)):
        trible_box = gen_boundbox(box, landmarks)
        tri_imgs = []
        for bbox in trible_box:
            bbox = bbox + padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            #cv2.imwrite("test.jpg", new_bd_img[w_min:w_max, h_min:h_max, :])
            tri_imgs.append([cv2.resize(new_bd_img[w_min:w_max, h_min:h_max, :], (64, 64))])

        for idx, pbox in enumerate(trible_box):
            pbox = pbox + padding
            h_min, w_min = pbox[0]
            h_max, w_max = pbox[1]
            new_bd_img = cv2.rectangle(new_bd_img, (h_min, w_min), (h_max, w_max), colors[idx], 2)

        image = np.array(tri_imgs,dtype=np.float32)
        img1 = image[0][0]
        img2 = image[1][0]
        img3 = image[2][0]

        img1.tofile('./output1/'+ file_name + '.bin')
        img2.tofile('./output2/'+ file_name + '.bin')
        img3.tofile('./output3/'+ file_name + '.bin')


path = './dataset/wiki_crop/'
for root,dirs,files in os.walk(path):
    for file in files:
        file_url = os.path.join(root,file)
        info = predict(cv2.imread(file_url), file)

print("=============转换完成=============")