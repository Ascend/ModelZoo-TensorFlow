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
import numpy as np
import pickle
import PIL.Image
import time
import dnnlib.tflib.tfutil as tfutil
import dnnlib.submission.submit as submit


# save_pkl, load_pkl are used by the mri code to save datasets
def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# save_snapshot, load_snapshot are used save/restore trained networks
def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)


def load_snapshot(fname):
    fname = os.path.join(submit.get_path_from_template(fname))
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_image(submit_config, img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if t.dtype in [np.float32, np.float64]:
        t = clip_to_uint8(t)
    else:
        assert t.dtype == np.uint8
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(submit_config.run_dir, filename))


def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def crop_np(img, x, y, w, h):
    return img[:, y:h, x:w]


# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    # x = x.transpose([1, 2, 0])
    return x.transpose([1, 2, 0])


# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return x.transpose([2, 0, 1])


# Run an image through the network (apply reflect padding when needed
# and crop back to original dimensions.)
def infer_image(net=None, img=None):
    w = img.shape[2]
    h = img.shape[1]
    pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
    padded_img = img
    if pw != 0 or ph != 0:
        padded_img = np.pad(img, ((0, 0), (0, ph), (0, pw)), 'reflect')

    # start = time.time()

    inferred = net.run(np.expand_dims(padded_img, axis=0), width=w + pw, height=h + ph)

    # time_run = time.time() - start
    # return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h)), time_run

    return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))


def infer_image_pb(input_tensor=None, output_tensor=None, img=None):
    inferred = tfutil.run([output_tensor], {input_tensor: np.expand_dims(img, axis=0)})
    return clip_to_uint8(inferred[0][0])
