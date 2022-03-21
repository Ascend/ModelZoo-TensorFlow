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
# -*- coding:utf-8 -*-
"""
将VOC2007和VOC2012图像数据集, 利用cv2.imwrite()函数保存图像时, quality factor为10, 20.
"""

import os
import cv2
import glob

if __name__ == '__main__':
    image_h = 256
    image_w = 256
    image_c = 1

    data_dir = "../datasets/BSDS300/images"

    save_dir = "../datasets/BSDS_Quality10"
    save_dir2 = "../datasets/BSDS_Quality100"
    print("go")
    for filename in ["train"]:
        image_path = glob.glob(os.path.join(data_dir, filename, "*.jpg"))
        # print(image_path[0].split("/")[8]) # Get the name of images.
        print(len(image_path))
        for i in range(len(image_path)):
            image = cv2.imread(image_path[i], 0) # Gray image
            image = cv2.resize(image,(image_w,image_h))

            save_path = os.path.join(save_dir, filename)
            save_path2 = os.path.join(save_dir2, filename)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path2):
                os.makedirs(save_path2)
            save_image_path = os.path.join(save_path, image_path[i].split("/")[-1])
            save_image_path2 = os.path.join(save_path2, image_path[i].split("/")[-1])
            cv2.imwrite(save_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) # quality 10.
            # default is 95.
            cv2.imwrite(save_image_path2, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(i)

