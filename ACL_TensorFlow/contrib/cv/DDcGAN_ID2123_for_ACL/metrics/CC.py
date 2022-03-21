"""License"""
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
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageStat
import math

# img2=Image.open('./metrics/F9_00.bmp')

def CC(fused, img):
    '''
    fused:fused image
    img:original image
    '''
    fused = fused.convert('L')
    img = img.convert('L')
    
    stat1=ImageStat.Stat(fused)
    stat2=ImageStat.Stat(img)


    img1=np.uint8(np.array(fused))
    img2=np.uint8(np.array(img))

    MN = img1.shape[0]*img1.shape[1]
    a = img1 - np.array(stat1.mean)
    b = img2 - np.array(stat2.mean)
    print(a.shape)
    print(b.shape)
    a[a < 0] =0
    b[b < 0] =0
    b[b > 0] +=1
    a = np.uint8(a)
    b = np.uint8(b)
    a = np.float64(a)
    b = np.float64(b)
    print(a.shape)
    print(b.shape)
    #c = sum(sum(a*b))
    #d = math.sqrt(sum(sum(a*a))*sum(sum(b*b)))
    ab = a * b
    ab[ab > 255] = 255
    aa = a * a 
    aa[aa > 255] = 255
    bb = b * b
    bb[bb > 255] = 255

    r = sum(sum(ab))/math.sqrt(sum(sum(aa))*sum(sum(bb)))
    # print(sum(sum(aa)))
    # print(sum(sum(bb)))
    # print(sum(sum(ab)))
    # print(bb)
    # print(r)
    # print(b)
    return r
# if __name__ == "__main__":
#     fused = Image.open("./fused/13.bmp")

#     img=Image.open('./ir/13.bmp')

 
#     print(CC(fused,img))




