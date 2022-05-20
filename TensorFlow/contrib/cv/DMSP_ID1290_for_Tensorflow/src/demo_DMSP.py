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


#npu代码
from npu_bridge.npu_init import *

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

#=========npu代码==============
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
#=============关闭融合策略=============
# custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/fusion_switch.cfg")
#==============打开混合精度============
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
#===============溢出检查================
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/cache/dump")
# custom_op.parameter_map["enable_dump_debug"].b = True
# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

# profiling
# custom_op.parameter_map["profiling_mode"].b = True
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/cache/profiling","task_trace":"on"}')
#=========npu代码==============
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)



#gpu代码
import os
def listdir(base_dir, list_name):  # 传入存储的list
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
#==============npu代码=========================
import argparse

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="./BSDS300/images/train")
# config = parser.parse_args()
config, unparsed = parser.parse_known_args()
# 在ModelArts容器创建数据存放目录
data_dir =data_url
# os.makedirs(data_dir)
# OBS数据拷贝到ModelArts容器内
# mox.file.copy_parallel(config.data_url, data_dir)
#========================================================================
list_name = []
base_dir = './train'  # 文件夹路径
listdir(data_dir, list_name)
size = len(list_name)
# Load data
sigma_d = 255 * .01
matFile = io.loadmat('./kernels.mat')
kernel = matFile['kernels'][0,0]
kernel = kernel / np.sum(kernel[:])

test_set = []
test_dir = data_dir
listdir(test_dir, test_set)
non_blind= []
i=1

# print("==================training DAE=====================")
# from DAE import DAE_MODEL
# dncnn = DAE_MODEL()
# dncnn.train(data_dir)

print("============start non-blind deblurring on Berkeley segmentation dataset==============")
params = {}
DAE = denoiser(sess)
params['denoiser'] = DAE
params['sigma_dae'] = 11.0
params['num_iter'] = 300
params['mu'] = 0.9
params['alpha'] = 0.1
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
    # img_degraded.save("/home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/data/degraded.png","png")

    # non-blind deblurring demo
    # run DMSP
    params['gt'] = gt # feed ground truth to monitor the PSNR at each iteration

    restored,psnr = DMSPDeblur(degraded, kernel, sigma_d, params)
    non_blind.append(psnr)
    img_restored = Image.fromarray(np.clip(restored, 0, 255).astype(dtype=np.uint8))
    # img_restored.save("/home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/data/restored.png","png")
    i+=1

print("non_blind PSNR is:",np.mean(non_blind))

print("============start noise-blind deblurring on Berkeley segmentation dataset==============")

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
    # img_restored_nb.save("/home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/restored_noise_blind.png","png")

print("noise-blind PSNR is:",np.mean(test_res))

print("========================")
# print(os.listdir(model_dir))
# print(os.listdir(profiling_dir))
print("========================")
# mox.file.copy_parallel(model_dir, "obs://train-dmsp/output/model/")
# mox.file.copy_parallel(profiling_dir, "obs://train-dmsp/output/profiling/")
