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

import numpy as np
import cv2

def equalize_and_stack_square (images, axis=1):
    max_c = max ([ 1 if len(image.shape) == 2 else image.shape[2]  for image in images ] )

    target_wh = 99999
    for i,image in enumerate(images):
        if len(image.shape) == 2:
            h,w = image.shape
            c = 1
        else:
            h,w,c = image.shape

        if h < target_wh:
            target_wh = h

        if w < target_wh:
            target_wh = w

    for i,image in enumerate(images):
        if len(image.shape) == 2:
            h,w = image.shape
            c = 1
        else:
            h,w,c = image.shape

        if c < max_c:
            if c == 1:
                if len(image.shape) == 2:
                    image = np.expand_dims ( image, -1 )
                image = np.concatenate ( (image,)*max_c, -1 )
            elif c == 2: #GA
                image = np.expand_dims ( image[...,0], -1 )
                image = np.concatenate ( (image,)*max_c, -1 )
            else:
                image = np.concatenate ( (image, np.ones((h,w,max_c - c))), -1 )

        if h != target_wh or w != target_wh:
            image = cv2.resize ( image, (target_wh, target_wh) )
            h,w,c = image.shape

        images[i] = image

    return np.concatenate ( images, axis = 1 )