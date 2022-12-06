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

from npu_bridge.npu_init import *
import cv2
import numpy as np
import random
from skimage import util


def add_noise(img):
    mode_types = ['gaussian', 'localvar', 'poisson', 'speckle']  # 'salt', 'pepper', 's&p' is too fake
    inx = int(np.random.choice(np.arange(len(mode_types)), 1))
    inx = 0
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    mean = random.random() * 0.001  # + 0.001#random.random() generates number between 0 and 1
    var = random.random() * 0.002  # + 0.01
    noise_img = util.random_noise(img.copy(), mode=mode_types[inx],
                                  mean=mean,
                                  var=var)
    return noise_img


def augment_data(img_patch, flip, rot): # img_patchs : n,h,w,c
    if flip==1:
        img_patch = img_patch[:, ::-1, :] # hflip
    elif flip==2:
        img_patch = img_patch[::-1, :, :] # vflip
    if rot==1:
        img_patch = cv2.rotate(img_patch, cv2.ROTATE_90_CLOCKWISE)
    elif rot==2:
        img_patch = cv2.rotate(img_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_patch

def preprocess(imgs, cfg):
    LR_patchs, HR_patchs = [], []
    for img in imgs:

        HR = cv2.imread(img.strip(), cv2.IMREAD_COLOR)
        HR = (HR - 127.5) / 128
        h, w, c = HR.shape

        x_stride = w // (cfg.imagesize * cfg.scale)
        y_stride = h // (cfg.imagesize * cfg.scale)

        for x in range(x_stride):
            for y in range(y_stride):
                HR_patch = HR[y * cfg.imagesize * cfg.scale:(y + 1) * cfg.imagesize * cfg.scale,
                           x * cfg.imagesize * cfg.scale:(x + 1) * cfg.imagesize * cfg.scale, :]
                # add noise && add blur
                t = np.random.randint(0, 2, 1)
                if t == 0:
                    LR_patch = cv2.resize(HR_patch, dsize=None, fx=1 / cfg.scale, fy=1 / cfg.scale,
                                          interpolation=cv2.INTER_CUBIC)
                    LR_patch = np.clip(LR_patch, -1.0, 1.0)
                    #LR_patch = add_noise(LR_patch)
                else:
                    #LR_patch = add_noise(HR_patch)  # [-1, 1]
                    LR_patch = cv2.resize(HR_patch, dsize=None, fx=1 / cfg.scale,
                                          fy=1 / cfg.scale, interpolation=cv2.INTER_LINEAR)
                # data augment
                if cfg.istrain:
                    rot = np.random.randint(0, 3, 1)
                    flip = np.random.randint(0, 3, 1)
                    LR_patch = augment_data(LR_patch, flip, rot)
                    HR_patch = augment_data(HR_patch, flip, rot)
                LR_patchs.append(LR_patch)
                HR_patchs.append(HR_patch)

    return HR_patchs, LR_patchs



