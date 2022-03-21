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
import cv2 as cv
import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil

import util
import config


class ValidationSet:
    def __init__(self, submit_config, height, width):
        self.images = None
        self.height = height
        self.width = width
        self.submit_config = submit_config
        return

    def resize_img(self, input_img, height, width):
        old_size = input_img.shape[0:2]
        target_size = [height, width]
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        img_new = cv.resize(input_img, (new_size[1], new_size[0]))
        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        resized_img = cv.copyMakeBorder(img_new, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        # resized_img = cv.resize(input_img, (args.width, args.height))
        return resized_img

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
                reshaped = self.resize_img(arr, height=self.height, width=self.width)
                reshaped = reshaped.transpose([2, 0, 1]) / 255.0 - 0.5
                images.append(reshaped)

            except OSError as e:
                print('Skipping file', fname, 'due to error: ', e)
        self.images = images

    def evaluate(self, input_tensor=None, output_tensor=None, iteration=None, noise_func=None):
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image_pb(input_tensor=input_tensor, output_tensor=output_tensor, img=noisy_img)
            orig255 = util.clip_to_uint8(orig_img)

            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255 * 255) / (s / (w * h * 3)))
            avg_psnr += cur_psnr

            util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                pass
                # util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                # util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print('Average PSNR: %.2f' % avg_psnr)


def validate_pb(submit_config: dnnlib.SubmitConfig, tf_config: dict, noise: dict, dataset: dict, pbdir: str, input_tensor_name: str,
                output_tensor_name: str, width: int, height: int):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config, height, width)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)
    graph = tfutil.load_pb(pbdir)
    tfutil.init_tf(tf_config, graph=graph)

    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    validation_set.evaluate(input_tensor=input_tensor, output_tensor=output_tensor, iteration=0,
                            noise_func=noise_augmenter.add_validation_noise_np)

    ctx.close()


def infer_image_pb(pbdir: str, input_tensor_name: str, output_tensor_name: str, image: str, out_image: str,
                   height: int, width: int, tf_config: dict):
    graph = tfutil.load_pb(pbdir)
    tfutil.init_tf(tf_config, graph=graph)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    validationSet = ValidationSet(None, None, None)
    arr = validationSet.resize_img(arr, height, width)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5

    input_tensor = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    pred255 = util.infer_image_pb(input_tensor, output_tensor, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print('Inferred image saved in', out_image)
