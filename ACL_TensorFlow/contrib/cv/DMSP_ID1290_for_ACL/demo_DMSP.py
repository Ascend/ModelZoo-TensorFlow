#!/usr/bin/env python 
# -*- coding:utf-8 -*-

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


import numpy as np
import scipy.io as io
from PIL import Image
import tensorflow as tf

# The DMSP deblur function and the RGB filtering function (flipped convolution)
# from DMSPDeblur import DMSPDeblur, filter_image

#2
from DMSPDeblur import DMSPDeblur, filter_image
# The denoiser implementation
from DAE_model import denoiser

# Limit the GPU access
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# configure the tensorflow and instantiate a DAE
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

DAE = denoiser(sess)


# Load data
sigma_d = 255 * .01
matFile = io.loadmat('kernels.mat')
kernel = matFile['kernels'][0,0]
kernel = kernel / np.sum(kernel[:])

#gpu代码
import os

def listdir(base_dir, list_name):  # 传入存储的list
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
# list_name = []
# base_dir = '/home/dataset/ILSVRC/Data/CLS-LOC/train'  # 文件夹路径
# listdir(base_dir, list_name)

test_set = []
test_dir = "/usr/model_test/dmsp/train"
listdir(test_dir, test_set)
non_blind= []
i=1
params = {}
params['denoiser'] = DAE
params['sigma_dae'] = 11.0
params['num_iter'] = 300
params['mu'] = 0.9
params['alpha'] = 0.1

print("============start non-blind deblurring on Berkeley segmentation dataset==============")
for image_path in test_set:
    if(i==51):
        break
    gt = Image.open(image_path)
    gt = gt.resize((180,180))
    gt = np.array(gt,dtype=np.float32)
    degraded = filter_image(gt, kernel)
    noise = np.random.normal(0.0, sigma_d, degraded.shape).astype(np.float32)
    degraded = degraded + noise

    img_degraded = Image.fromarray(np.clip(degraded, 0, 255).astype(dtype=np.uint8))
    img_degraded.save("data/degraded.png","png")

    # non-blind deblurring demo
    # run DMSP

    # gt = Image.open(r'/home/dataset/Set14/baboon.png')
    params['gt'] = gt # feed ground truth to monitor the PSNR at each iteration

    restored,psnr = DMSPDeblur(degraded, kernel, sigma_d, params)
    non_blind.append(psnr)
    img_restored = Image.fromarray(np.clip(restored, 0, 255).astype(dtype=np.uint8))
    img_restored.save("data/restored.png","png")
    i+=1
print("non_blind PSNR is:",np.mean(non_blind))

print("============start noise-blind deblurring on Berkeley segmentation dataset==============")
# noise-blind deblurring demo
# run DMSP noise-blind
# params = {}
# params['denoiser'] = DAE
# params['sigma_dae'] = 11.0
# params['num_iter'] = 300
# params['mu'] = 0.9
# params['alpha'] = 0.1


test_res = []

j = 0
for test_image in test_set:
    j+=1
    if(j==51):
        break
    gt = Image.open(test_image)
    gt = gt.resize((180, 180))
    # nshape = np.array(gt, dtype=np.float32)
    # gt = gt.resize((int(nshape.shape[1]),int(nshape.shape[0])))
    gt = np.array(gt, dtype=np.float32)
    degraded = filter_image(gt, kernel)
    params['gt'] = gt

    noise = np.random.normal(0.0, sigma_d, degraded.shape).astype(np.float32)

    degraded = degraded + noise
    img_degraded = Image.fromarray(np.clip(degraded, 0, 255).astype(dtype=np.uint8))

    restored_nb,psnr_test = DMSPDeblur(degraded, kernel, -1, params)
    test_res.append(psnr_test)
    img_restored_nb = Image.fromarray(np.clip(restored_nb, 0, 255).astype(dtype=np.uint8))
    img_restored_nb.save("data/restored_noise_blind.png","png")

print("noise-blind PSNR is:",np.mean(test_res))