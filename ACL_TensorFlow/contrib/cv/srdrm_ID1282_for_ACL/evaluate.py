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
# @Time    : 2021/10/24 14:20
# @Author  : XJTU-zzf
# @FileName: evaluate.py
"""

from utils.data_utils import getPaths
from utils.uqim_utils import getUIQM
from utils.ssim_psnr_utils import getSSIM, getPSNR
from PIL import Image
import numpy as np
import ntpath
import os


# mesures uqim for all images in a directory
def measure_UIQMs(dir_name):
    """
       param: dir_name
       return: uqim
    """
    paths = getPaths(dir_name)
    uqims = []
    for img_path in paths:
        # im_ = misc.imread(img_path)
        im = np.array(Image.open(img_path))
        uqims.append(getUIQM(im))
    return np.array(uqims)


# compares avg ssim and psnr
def measure_SSIM_PSNRs(GT_dir, Gen_dir):
    """
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_gen.jpg}
        * Images are of same-size
    """
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)
    ssims, psnrs = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0] + '_gen.jpg')  # +name_split[1])
        if gen_path in Gen_paths:
            r_im = np.array(Image.open(img_path).convert('L'))
            g_im = np.array(Image.open(gen_path).convert('L'))
            # r_im = misc.imread(img_path, mode="L")
            # g_im = misc.imread(gen_path, mode="L")
            assert (r_im.shape == g_im.shape), "The images should be of same-size"
            ssim = getSSIM(r_im, g_im)
            psnr = getPSNR(r_im, g_im)
            ssims.append(ssim)
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)
