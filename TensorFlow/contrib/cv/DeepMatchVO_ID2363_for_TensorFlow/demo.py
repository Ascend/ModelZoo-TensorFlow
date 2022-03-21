# -*- coding=utf-8 -*-
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
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import matplotlib.pyplot as plt
from deep_slam import DeepSlam

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    return depth

if __name__ == '__main__':
    img_height=128
    img_width=416
    ckpt_file = 'ckpt/model-250000'
    fh = open('data/example.png', 'r')
    # fh = open('data/example.png', 'r')################wty
    I = pil.open('data/example.png')
    I = I.resize((img_width, img_height), pil.ANTIALIAS)
    I = np.array(I)
    
    system = DeepSlam()
    system.setup_inference(img_height, img_width, mode='depth')
    
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    with tf.Session(config=npu_config_proto()) as sess:
        tf.get_variable_scope().reuse_variables()########wty
        saver.restore(sess, ckpt_file)
        pred = system.inference(sess, mode='depth', inputs=I[None,:,:,:])
    
    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    plt.imshow(I)
    plt.subplot(2,2,2)
    plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    plt.show()
    
