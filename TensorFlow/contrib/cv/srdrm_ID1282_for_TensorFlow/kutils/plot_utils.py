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
# > Various modules for plots 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
import os
import matplotlib.pyplot as plt


def sampleImages(imgs_lr, fake_hr, imgs_hr, dir_, step):
    # Rescale images 0 - 1
    imgs_lr = 0.5 * imgs_lr + 0.5
    fake_hr = 0.5 * fake_hr + 0.5
    imgs_hr = 0.5 * imgs_hr + 0.5
    r, c = 2, 2
    # Save generated images and the high resolution originals
    titles = ['Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for row in range(r):
        for col, image in enumerate([fake_hr, imgs_hr]):
            axs[row, col].imshow(image[row])
            axs[row, col].set_title(titles[col])
            axs[row, col].axis('off')
        cnt += 1
    fig.savefig(os.path.join(dir_, ("%d.png" % step)))
    plt.close()
    # Save low resolution images for comparison
    for i in range(r):
        fig = plt.figure()
        plt.imshow(imgs_lr[i])
        fig.savefig(os.path.join(dir_, ("%d_lowres_%d.png" % (step, i))))
        plt.close()


def save_val_samples(samples_dir, gen_imgs, step, N_samples=2, N_ims=2):
    row = N_samples
    col = N_ims
    titles = ['Generated', 'Original']
    fig, axs = plt.subplots(row, col)
    cnt = 0
    for j in range(col):
        for i in range(row):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(samples_dir, ("%d.png" % step)))
    plt.close()
