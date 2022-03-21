"""
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
# @Time    : 2021/10/22 14:25
# @Author  : XJTU-zzf
# @FileName: img_preprocess.py
"""

import argparse
import os
import ntpath
import numpy as np
from utils.data_utils import getPaths, preprocess
from PIL import Image


def pre_process_img(img_path, lr_shape):
    """
    param: img_path, lr_shape
    return: img numpy
    """
    img_lr = Image.open(img_path).convert('RGB')
    img_lr = np.array(img_lr.resize((lr_shape[1], lr_shape[0])), dtype="float32")

    im = preprocess(img_lr)
    im = np.expand_dims(im, axis=0)

    return im


def mode2shape(data_mode):
    """
    param: data_mode String
    return: tuple (img_size)
    """
    if data_mode == '2x':
        shape = (240, 320)
    elif data_mode == '4x':
        shape = (120, 160)
    else:
        shape = (60, 80)
    return shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/data/wind/dataset/SRDRM/USR248/TEST/")
    parser.add_argument("--data_mode", default="8x")
    parser.add_argument("--dst_path", default="./input_bz32", help="path of output bin files")
    parser.add_argument("--pic_num", default=-1, help="picture number")
    parser.add_argument("--batch_size", default=32)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, 'lr_{}'.format(args.data_mode))
    lr_shape = mode2shape(args.data_mode)
    dst_path = args.dst_path
    pic_num = args.pic_num
    bz = int(args.batch_size)

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    test_paths = getPaths(data_dir)

    n = 0
    i = 0
    while i < len(test_paths):
        temp = []
        name = []

        if i + bz > len(test_paths):
            break

        for j in range(0, bz, 1):
            print("start to process %s" % test_paths[i + j])
            img_name = ntpath.basename(test_paths[i + j]).split('.')[0]
            res = pre_process_img(test_paths[i + j], lr_shape)
            temp.append(res)
            name.append(img_name)

        img_bz = np.concatenate(temp, axis=0)
        bin_name = ""
        for img_name in name:
            bin_name += img_name.split("_")[2] + "_"
        img_bz.tofile(dst_path + "/" + bin_name + ".bin")
        i = i + bz
        n += bz
        if int(pic_num) == n:
            break

    print("共生成bin文件: {}.".format(len(os.listdir(dst_path))))
