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
from __future__ import division
from npu_bridge.npu_init import *
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import os
import argparse
import imageio
import matplotlib.pylab as plt
import cv2
import skimage
import json

FLAGS = tf.flags.FLAGS

# adding some parameters
tf.flags.DEFINE_string('input_dir', './dataset/Sony/short/', "input dir.")
tf.flags.DEFINE_string('gt_dir', './dataset/Sony/long/', "gt dir.")


# get test IDs
test_fns = glob.glob(FLAGS.gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

ps = 512  # patch size for training #512
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(test_ids)
input_images['250'] = [None] * len(test_ids)
input_images['100'] = [None] * len(test_ids)

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(FLAGS.input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(FLAGS.gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_full.shape[1]
        W = input_full.shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)

        input_patch = input_full[:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_full[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))


        #input_patch = np.minimum(input_patch, 1.0)
        input_full = np.minimum(input_full, 1.0)
        # print(np.shape(input_patch))
        fileddata = np.ascontiguousarray(input_full)
        with open('./bin_file/' + in_path + '.bin', "wb") as f:
            f.write(fileddata)


        fileddata_gt = np.ascontiguousarray(gt_full)
        with open('./bin_file/' + gt_path + '.bin', "wb") as f:
            f.write(fileddata_gt)