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

import argparse
import os

import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import structural_similarity


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-ddm', '--ddm_sigmoid3', required=True, type=str, \
                   help='folder of input image or file of one input.')
    p.add_argument('-sd', '--sd_sigmoid4', required=True, type=str, \
                   help='folder of input image or file of one input.')
    p.add_argument('-sdm', '--sdm_sigmoid5', required=True, type=str, \
                   help='folder of input image or file of one input.')
    p.add_argument('-dst', '--dst', required=True, type=str, \
                   help='folder of input image or file of one input.')
    return p.parse_args()


def ssim(image1, image2):
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image1 = rgb2ycbcr(image1)[:, :, 0:1]
    image2 = rgb2ycbcr(image2)[:, :, 0:1]
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    ssim_val = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True,
                                     data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
    return ssim_val


def cal_ssim(args, ddm_sigmoid3, sd_sigmoid4, sdm_sigmoid5, dst):
    ddm_sigmoid3 = os.path.join(args.ddm_sigmoid3, ddm_sigmoid3)
    ddm_sigmoid3 = np.fromfile(ddm_sigmoid3, dtype=np.float32)
    ddm_sigmoid3 = np.reshape(ddm_sigmoid3, (96, 96, 1))
    ddm_sigmoid3 = np.repeat(ddm_sigmoid3, (3,), -1)

    sd_sigmoid4 = np.fromfile(sd_sigmoid4, dtype=np.float32)
    sd_sigmoid4 = np.reshape(sd_sigmoid4, (96, 96, 3))

    sdm_sigmoid5 = np.fromfile(sdm_sigmoid5, dtype=np.float32)
    sdm_sigmoid5 = np.reshape(sdm_sigmoid5, (96, 96, 1))
    sdm_sigmoid5 = np.repeat(sdm_sigmoid5, (3,), -1)

    dst = np.fromfile(dst, dtype=np.float32)
    dst = np.reshape(dst, (96, 96, 3))

    dst_masked = (dst * ddm_sigmoid3).astype(np.uint8)
    predict_masked = (sd_sigmoid4 * (ddm_sigmoid3 * sdm_sigmoid5) * 255).astype(np.uint8)
    return ssim(dst_masked, predict_masked)


def main():
    args = get_args()
    ddm_bin_files = os.listdir(args.ddm_sigmoid3)
    ssim_list = []
    for ddm_bin_file in ddm_bin_files:
        sd_bin_file = os.path.join(args.sd_sigmoid4, ddm_bin_file)
        sdm_bin_file = os.path.join(args.sdm_sigmoid5, ddm_bin_file)
        dst = os.path.join(args.dst, (os.path.splitext(os.path.split(ddm_bin_file)[1])[0][:7] + ".bin"))
        ssim_result = cal_ssim(args, ddm_bin_file, sd_bin_file, sdm_bin_file, dst)
        ssim_list.append(ssim_result)
    ssim_list = np.asarray(ssim_list)
    print(f"[Info] SSIM: {(int(ssim_list.mean()*100))/100}")


if __name__ == '__main__':
    main()
