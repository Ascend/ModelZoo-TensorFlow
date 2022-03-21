# -*- coding: utf-8 -*-
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
"""
Created on Sun Feb 24 16:31:16 2019

@author: wmy
"""
from npu_bridge.npu_init import *

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

original_images_path = r'./datasets/OriginalImages'
color_images_path = r'./datasets/ColorImages'
grayscale_images_path = r'./datasets/GrayscaleImages'
combined_images_path = r'./datasets/CombinedImages'

resize_height = 256
resize_weidth = 256

def find_images(path):
    result = []
    for filename in os.listdir(path):
        _, ext = os.path.splitext(filename.lower())
        if ext == ".jpg" or ext == ".png":
            result.append(os.path.join(path, filename))
            pass
        pass
    result.sort()
    return result

if __name__ == '__main__':
    search_result = find_images(original_images_path)
    for image_path in search_result:
        img_name = image_path[len(original_images_path):]
        img = Image.open(image_path)
        img_color = img.resize((resize_weidth, resize_height), Image.ANTIALIAS)
        img_color.save(color_images_path + img_name, quality=95)
        print("Info: image '" + color_images_path + img_name + "' saved.")
        img_gray = img_color.convert('L')
        img_gray = img_gray.convert('RGB')
        img_gray.save(grayscale_images_path + img_name, quality=95)
        print("Info: image '" + grayscale_images_path + img_name + "' saved.")
        combined_image = Image.new('RGB', (resize_weidth*2, resize_height))
        combined_image.paste(img_color, (0, 0, resize_weidth, resize_height))
        combined_image.paste(img_gray, (resize_weidth, 0, resize_weidth*2, resize_height))
        combined_image.save(combined_images_path + img_name, quality=95)
        print("Info: image '" + combined_images_path + img_name + "' saved.")
        pass
