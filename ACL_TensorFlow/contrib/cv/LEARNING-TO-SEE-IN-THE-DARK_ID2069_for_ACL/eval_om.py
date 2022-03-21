#
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
#
from npu_bridge.npu_init import *
import numpy as np
import math
import cv2
import tensorflow as tf
import skimage
import glob
import os, scipy.io
#from skimage.measure import compare_ssim
from skimage import measure
from skimage.metrics import structural_similarity

FLAGS = tf.flags.FLAGS

# adding some parameters
tf.flags.DEFINE_string('gt_dir', './dataset/Sony/long/', "gt dir.")
tf.flags.DEFINE_string('pic_dir', './bin_file/dataset/Sony/', "pic dir.")

test_fns = glob.glob(FLAGS.gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

def psnr(img1, img2): #1-out 2-gt
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)
    diff = img2 - img1
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    return 20*math.log10(255.0/rmse)

a = []
b = []

for test_id in test_ids:
    pic_gts_10s = FLAGS.pic_dir + 'long/%05d_00_10s.ARW.bin' % test_id
    pic_gts_30s = FLAGS.pic_dir + 'long/%05d_00_30s.ARW.bin' % test_id
    pic_0_1_outs = FLAGS.pic_dir + 'om_out/%05d_00_0.1s.ARW_output_0.bin' % test_id
    pic_0_04_outs = FLAGS.pic_dir + 'om_out/%05d_00_0.04s.ARW_output_0.bin' % test_id
    pic_0_033_outs = FLAGS.pic_dir + 'om_out/%05d_00_0.03s.ARW_output_0.bin' % test_id


    if os.path.exists(pic_0_1_outs):
        #img_out_all = np.fromfile(pic_0_1_outs, np.float32).reshape(1, 1024, 1024, 3)
        img_out_all = np.fromfile(pic_0_1_outs, np.float32).reshape(1, 2848, 4256, 3)
        img_out_all = np.minimum(np.maximum(img_out_all, 0), 1)
        img_out = (img_out_all[0, :, :, :] * 255).astype(np.uint8)
        if(os.path.exists(pic_gts_10s)):
            #img_gt_all = np.fromfile(pic_gts_10s, np.float32).reshape(1, 1024, 1024, 3)
            img_gt_all = np.fromfile(pic_gts_10s, np.float32).reshape(1, 2848, 4256, 3)
        else:
            #img_gt_all = np.fromfile(pic_gts_30s, np.float32).reshape(1, 1024, 1024, 3)
            img_gt_all = np.fromfile(pic_gts_30s, np.float32).reshape(1, 2848, 4256, 3)


        img_gt = (img_gt_all[0, :, :, :] * 255).astype(np.uint8)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)

    if os.path.exists(pic_0_04_outs):
        img_out_all = np.fromfile(pic_0_04_outs, np.float32).reshape(1, 2848, 4256, 3)
        img_out_all = np.minimum(np.maximum(img_out_all, 0), 1)
        img_out = (img_out_all[0, :, :, :] * 255).astype(np.uint8)
        if (os.path.exists(pic_gts_10s)):
            img_gt_all = np.fromfile(pic_gts_10s, np.float32).reshape(1, 2848, 4256, 3)
        else:
            img_gt_all = np.fromfile(pic_gts_30s, np.float32).reshape(1, 2848, 4256, 3)

        img_gt = (img_gt_all[0, :, :, :] * 255).astype(np.uint8)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)

    if os.path.exists(pic_0_033_outs):
        img_out_all = np.fromfile(pic_0_033_outs, np.float32).reshape(1, 2848, 4256, 3)
        img_out_all = np.minimum(np.maximum(img_out_all, 0), 1)
        img_out = (img_out_all[0, :, :, :] * 255).astype(np.uint8)
        if (os.path.exists(pic_gts_10s)):
            img_gt_all = np.fromfile(pic_gts_10s, np.float32).reshape(1, 2848, 4256, 3)
        else:
            img_gt_all = np.fromfile(pic_gts_30s, np.float32).reshape(1, 2848, 4256, 3)

        img_gt = (img_gt_all[0, :, :, :] * 255).astype(np.uint8)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)

print(a)
print(b)
average_psnr = sum(a)/len(a)
average_ssim = sum(b)/len(b)

print("average psnr: ", average_psnr)
print("average ssim: ", average_ssim)



