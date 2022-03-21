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
# @Time    : 2021/10/22 15:15
# @Author  : XJTU-zzf
# @FileName: bin2img_measure.py
"""

from utils.data_utils import deprocess
import numpy as np
from PIL import Image
import os
import argparse
from evaluate import measure_SSIM_PSNRs, measure_UIQMs

# 离线推理的出来的最终结果数据范围有问题,但是对最终的指标无影响


def deprocess_save(img, img_name, output):
    """
    param: img, img_name
           output: 图像的输出路径
    return: None
    """
    img = deprocess(img)  # Rescale to 0-1
    # save sample images
    output = os.path.join(output, img_name + '_gen.jpg')
    img = np.uint8(img * 255)
    Image.fromarray(img).convert("RGB").save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input", default="", help="离线推理结果bin文件")
    parser.add_argument("--dst_path", default="./bin2img_bz16", help="path of output bin files")
    parser.add_argument("--gt_dir", default="/mnt/data/wind/dataset/SRDRM/USR248/TEST/hr")
    parser.add_argument("--batch_size", default="16")
    args = parser.parse_args()
    print(args)

    data_input = args.data_input
    data_output = args.dst_path
    gt_dir = args.gt_dir
    bz = int(args.batch_size)

    if not os.path.exists(data_output):
        os.makedirs(data_output)
    # 还原图像
    files = os.listdir(data_input)

    for file in files:
        if file.endswith(".bin"):
            img = np.fromfile(data_input + '/' + file, dtype="float32")
            img = img.reshape(bz, 480, 640, 3)  # (1, 480, 640, 3)
            img_name_e = file.split("_")
            for i in range(0, bz):
                img_name = "im_xb_" + img_name_e[i] + "_"
                deprocess_save(img[i], img_name, data_output)

    # 评估
    SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(gt_dir, data_output)
    print("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
    print("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
    gen_uqims = measure_UIQMs(data_output)
    print("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

