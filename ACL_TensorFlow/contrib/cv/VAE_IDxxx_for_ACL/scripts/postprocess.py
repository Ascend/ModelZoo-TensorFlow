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
import numpy as np
import argparse
import sys
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt
import imageio
ims = imageio.imsave

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

if __name__ == '__main__':
    pred_dir = sys.argv[1]
    pred = np.fromfile(pred_dir, np.float32)
    generated_test = pred.reshape(100,28,28)
    ims("results_images/result.jpg",merge(generated_test[:64],[8,8]))
    print(pred)