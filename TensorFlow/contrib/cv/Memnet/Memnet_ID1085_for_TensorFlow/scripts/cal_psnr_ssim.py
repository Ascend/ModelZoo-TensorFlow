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
# -*- coding:utf-8 -*-
"""
Compute PSNR and SSIM with Set12.
"""
from npu_bridge.npu_init import *
import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity,peak_signal_noise_ratio


if __name__ == '__main__':
    data_set12 = glob.glob(os.path.join("../datasets/Set12", "*.jpg"))
    data_set2_quality10 = glob.glob(os.path.join("../datasets/Set12_Quality10", "*.jpg"))
    data_set12_recovery = glob.glob(os.path.join("../datasets/Set12_Recovery", "*.jpg"))

    compress_avg_psnr = 0.
    deblocking_avg_psnr = 0.
    compress_avg_ssim = 0.
    deblocking_avg_ssim = 0.
    for i in range(len(data_set12)):
        print("image:"+str(i))
        # reszie 256 * 256.
        img_set12 = cv2.resize(cv2.imread(str(data_set12[i]), 0), (256, 256))
        img_set12_q10 = cv2.resize(cv2.imread(str(data_set2_quality10[i]), 0), (256, 256))
        img_set12_recovery = cv2.resize(cv2.imread(str(data_set12_recovery[i]), 0), (256, 256))

        # label, noisy_image
        psnr_compress = peak_signal_noise_ratio(img_set12, img_set12_q10, data_range=255)
        print(psnr_compress)
        compress_avg_psnr += psnr_compress
        psnr_deblocking = peak_signal_noise_ratio(img_set12, img_set12_recovery, data_range=255)
        print(psnr_deblocking)
        deblocking_avg_psnr += psnr_deblocking

        ssim_compress = structural_similarity(img_set12, img_set12_q10)
        print(ssim_compress)
        compress_avg_ssim += ssim_compress
        ssim_deblocking = structural_similarity(img_set12, img_set12_recovery)
        print(ssim_deblocking)
        deblocking_avg_ssim += ssim_deblocking

    print("Average compress PSNR is: {}".format(compress_avg_psnr / len(data_set12)))
    print("Average compress SSIM is: {}".format(compress_avg_ssim / len(data_set12)))

    print("Average deblocking PSNR is: {}".format(deblocking_avg_psnr / len(data_set12)))
    print("Average deblocking SSIM is: {}".format(deblocking_avg_ssim / len(data_set12)))
