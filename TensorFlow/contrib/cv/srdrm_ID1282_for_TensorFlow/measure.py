#!/usr/bin/env python
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
"""
# > Script for measuring quantitative performances in terms of
#    - Underwater Image Quality Measure (UIQM)
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
# python libs
import os
import ntpath
import numpy as np
# local libs
from kutils.data_utils import getPaths
from kutils.uqim_utils import getUIQM
from kutils.ssm_psnr_utils import getSSIM, getPSNR
from PIL import Image
import tensorflow as tf
import datetime
# logger
from kutils import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.flags
flags.DEFINE_string(name='measure_mode', default='8x', help='choice=[2x, 4x, 8x]')
flags.DEFINE_string(name='chip', default='gpu', help='choice=[gpu, npu, cpu]')
flags.DEFINE_string(name="gt_dir", default='/mnt/data/wind/dataset/SRDRM/USR248/TEST/hr', help='ground truth data_dir')
flags.DEFINE_string(name="gen_dir_prefix", default='/mnt/data/wind/SRDRM/', help='gen_data_dir')
flags.DEFINE_string(name="obs_dir", default="obs://srdrm/", help="obs result path, not need on gpu and apulis platform")
flags.DEFINE_string(name='result', default='', help='使用NPU时需要用到，其余训练状态下为空')
flags.DEFINE_boolean(name="profiling", default=False, help="profiling for performance or not")
flags.DEFINE_string(name='model_name', default="srdrm", help="choice=[srdrm, srdrm-gan]")
flags.DEFINE_string(name='output', default='', help='输出路径，只在在modelarts上训练时要用')
flags.DEFINE_string(name='test_epoch', default='20', help='要measure的epoch')
flags.DEFINE_string(name="platform", default="linux",
                    help="Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy operations")
Flags = flags.FLAGS

# data paths
GTr_im_dir = Flags.gt_dir  # ground truth im-dir with {f.ext}
GEN_im_dir = os.path.join(Flags.gen_dir_prefix, '{}_test_output'.format(Flags.chip),
                          'epoch_{}'.format(Flags.test_epoch),
                          'USR_{}'.format(Flags.measure_mode),
                          Flags.model_name)  # generated im-dir with {f_gen.png}

# 记录控制台的输出的日志文件
logging_file = os.path.join(Flags.output, "task_log/measure/{}/USR_{}/{}".format(Flags.chip, Flags.measure_mode, Flags.model_name))
if not os.path.exists(logging_file):
    os.makedirs(logging_file)
log = Logger.Log(os.path.join(logging_file, "measure_epoch_{}.log".format(Flags.test_epoch)))

log.info(GEN_im_dir)
assert os.path.exists(GEN_im_dir) or len(os.listdir(GEN_im_dir)) == 0, "生成的图片不存在"
GEN_paths = getPaths(GEN_im_dir)


# mesures uqim for all images in a directory
def measure_UIQMs(dir_name):
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


# compute SSIM and PSNR
SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(GTr_im_dir, GEN_im_dir)
log.info("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
log.info("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

# compute and compare UIQMs
# g_truth = measure_UIQMs(GTr_im_dir)
# log.info ("G. Truth UQIM  >> Mean: {0} std: {1}".format(np.mean(g_truth), np.std(g_truth)))
gen_uqims = measure_UIQMs(GEN_im_dir)
log.info("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

if Flags.platform.lower() == 'modelarts':
    from help_modelarts import modelarts_result2obs
    modelarts_result2obs(Flags)
