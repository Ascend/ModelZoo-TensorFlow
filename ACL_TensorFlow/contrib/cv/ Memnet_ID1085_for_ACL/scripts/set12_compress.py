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
from npu_bridge.npu_init import *
import os
import cv2
import glob

if __name__ == '__main__':
    data_dir = "../dataset/Set12"

    save_path = "../datasets/Set12_Quality10"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image_file in glob.glob(os.path.join(data_dir, "*.png")):
        image = cv2.imread(image_file, 0) # Gray image

        save_image_path = os.path.join(save_path, image_file.split("/")[7][0:2] + ".jpg") # 01.jpg
        cv2.imwrite(save_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 10]) # quality 10. 
        # default is 95.
        # cv2.imwrite(save_image_path, image)
