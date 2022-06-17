#!/usr/bin/env python
# coding=utf-8

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
# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import cv2
import argparse
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./dataset/scenery_photo', type=str)
    parser.add_argument("--save_folder", default='./input_bin', type=str)

    args = parser.parse_args()

    return args


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder):

    name_list = os.listdir(load_folder)
    for name in name_list:
        load_path = os.path.join(load_folder, name)
        save_path = os.path.join(save_folder, name+ ".bin")
        image = cv2.imread(load_path)
        image = resize_crop(image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0).reshape(-1)
        batch_image.tofile(save_path)
        file = np.fromfile(save_path, dtype=np.float32)

if __name__ == '__main__':
    args = arg_parser()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    cartoonize(args.data_path, args.save_folder)
