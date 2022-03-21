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
from npu_bridge.npu_init import *
import os
import cv2
import glob

if __name__ == '__main__':
    data_dir = "./VOC0712"
    save_dir = "./datasets/VOC0712"

    for filename in ["VOC2007", "VOC2012"]:
        image_path = glob.glob(os.path.join(data_dir, filename, "JPEGImages/*.jpg"))
        # print(image_path[0].split("/")[8]) # Get the name of images.

        for i in range(len(image_path)):
            image = cv2.imread(image_path[i], 0) # Gray image

            save_path = os.path.join(save_dir, filename)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_image_path = os.path.join(save_path, image_path[i].split("/")[8])
            # cv2.imwrite(save_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) # quality 10. 
            # default is 95.
            cv2.imwrite(save_image_path, image)

