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
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
from npu_bridge.npu_init import *
import math
import pprint
import scipy.misc
import numpy as np
import copy
from PIL import Image
import imageio
from skimage import img_as_ubyte
from imageio import imread as _imread
# try:
#     _imread = scipy.misc.imread
# except AttributeError:
#     from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_test_data(image_path, fine_size=256, is_grayscale=True):
    img = imread(image_path, is_grayscale=is_grayscale)
    img = np.array(Image.fromarray(img).resize([fine_size, fine_size]))
    img = img / 127.5 - 1
    return img


def load_train_data(image_path, load_size=256, fine_size=256, is_testing=False, is_grayscale=True):
    img_A = imread(image_path[0], is_grayscale=is_grayscale)
    img_B = imread(image_path[1], is_grayscale=is_grayscale)
    if not is_testing:
        img_A = np.array(Image.fromarray(img_A).resize([load_size, load_size]))
        img_B = np.array(Image.fromarray(img_B).resize([load_size, load_size]))
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_A = img_A[h1:h1 + fine_size, w1:w1 + fine_size]
        img_B = img_B[h1:h1 + fine_size, w1:w1 + fine_size]
        # 预处理:1.load:grayscale 2.resize(256,256)
        if np.random.random() > 0.5: #随机
            img_A = np.fliplr(img_A) #在左右方向上翻转每行的元素，列保持不变，但是列的显示顺序变了
            img_B = np.fliplr(img_B)
    else:
        img_A = np.array(Image.fromarray(img_A).resize([fine_size, fine_size]))
        img_B = np.array(Image.fromarray(img_B).resize([fine_size, fine_size]))

    img_A = img_A / 127.5 - 1.  #??
    img_B = img_B / 127.5 - 1.
    if is_grayscale:
        img_A = np.reshape(img_A, newshape=(fine_size, fine_size, 1))
        img_B = np.reshape(img_B, newshape=(fine_size, fine_size, 1))
    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if is_grayscale:
        return _imread(path, as_gray=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return imageio.imwrite(path, img_as_ubyte(merge(images, size)))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return np.array(Image.fromarray(x[j:j + crop_h, i:i + crop_w]).resize([resize_h, resize_w]))


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def boolean_string(bool_str):
    bool_str = bool_str.lower()

    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")

    return bool_str == "true"

