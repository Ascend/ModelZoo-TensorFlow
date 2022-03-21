# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# ============================================================================
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

import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf
from npu_bridge.npu_init import *
import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util
import config


class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob

        abs_dirname = os.path.join(submit.get_path_from_template(dataset_dir), '*')
        fnames = sorted(glob.glob(abs_dirname))
        if len(fnames) == 0:
            print('\nERROR: No files found using the following glob pattern:', abs_dirname, '\n')
            sys.exit(1)

        images = []
        for fname in fnames:
            try:
                im = PIL.Image.open(fname).convert('RGB')
                arr = np.array(im, dtype=np.float32)
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                images.append(reshaped)
            except OSError as e:
                print('Skipping file', fname, 'due to error: ', e)
        self.images = images

    def evaluate(self, net, iteration, noise_func):
        avg_psnr = 0.0

        # time_all = 0
        # n = 0

        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image(net, noisy_img)

            # pred255, time_run = util.infer_image(net, noisy_img)
            # print(time_run)

            orig255 = util.clip_to_uint8(orig_img)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255 * 255) / (s / (w * h * 3)))
            avg_psnr += cur_psnr

            # util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            # if idx > 3:
            #     time_all += time_run
            #     n += 1

            if iteration == 0:
                pass
                # util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                # util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)

        # avg_sec = n / time_all

        with npu_scope.without_npu_compile_scope():
            print('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))

        # print('Validate average time images/sec: {}'.format(avg_sec))


def validate(submit_config: dnnlib.SubmitConfig, tf_config: dict, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)

    tfutil.init_tf(tf_config)

    with tf.device(None):
        net = util.load_snapshot(network_snapshot)
        validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)
    ctx.close()


def infer_image(network_snapshot: str, image: str, out_image: str, tf_config: dict):
    tfutil.init_tf(tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print('Inferred image saved in', out_image)
