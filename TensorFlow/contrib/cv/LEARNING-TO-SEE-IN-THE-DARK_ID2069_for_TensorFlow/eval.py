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
import os.path


os.environ["JOB_ID"] = "1231321312312323"


FLAGS = tf.flags.FLAGS

# adding some parameters
tf.flags.DEFINE_string('gt_dir', './dataset/Sony/long/', "gt dir.")
tf.flags.DEFINE_string('pic_dir', './result_Sony/final/', "pic dir.")

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
    pic_300_outs = FLAGS.pic_dir + '%05d_00_300_out.png' % test_id
    pic_300_gts = FLAGS.pic_dir + '%05d_00_300_gt.png' % test_id
    pic_250_outs = FLAGS.pic_dir + '%05d_00_250_out.png' % test_id
    pic_250_gts = FLAGS.pic_dir + '%05d_00_250_gt.png' % test_id
    pic_100_outs = FLAGS.pic_dir + '%05d_00_100_out.png' % test_id
    pic_100_gts = FLAGS.pic_dir + '%05d_00_100_gt.png' % test_id


    if os.path.exists(pic_300_outs):
        img_out = cv2.imread(pic_300_outs)
        img_gt = cv2.imread(pic_300_gts)
        print(img_out.dtype)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)

    if os.path.exists(pic_250_outs):
        img_out = cv2.imread(pic_250_outs)
        img_gt = cv2.imread(pic_250_gts)
        print(img_out.dtype)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)

    if os.path.exists(pic_100_outs):
        img_out = cv2.imread(pic_100_outs)
        img_gt = cv2.imread(pic_100_gts)
        print(img_out.dtype)
        p = psnr(img_out, img_gt)
        SSIM = structural_similarity(img_out, img_gt, multichannel=True)
        a = np.append(a, p)
        b = np.append(b, SSIM)


average_psnr = sum(a)/len(a)
average_ssim = sum(b)/len(b)

print("average psnr: ", average_psnr)
print("average ssim: ", average_ssim)
