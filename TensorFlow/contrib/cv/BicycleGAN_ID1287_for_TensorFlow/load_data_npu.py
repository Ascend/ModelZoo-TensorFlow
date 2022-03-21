
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
import os
import h5py
import glob
# import scipy.misc
# from scipy.misc import imread, imresize
from PIL import Image
# import moxing as mox
import matplotlib.image as mp

imread = Image.open


def load_images(maps_dir):
    train_all = glob.glob(maps_dir + "/train/*.jpg")
    train_img_A = []
    train_img_B = []

    for file in train_all:
        full_image = np.array(imread(file).resize((512, 256), Image.ANTIALIAS))
        img_B = full_image[:, full_image.shape[1] // 2:, :]
        img_A = full_image[:, :full_image.shape[1] // 2, :]
        train_img_A.append(img_A)
        train_img_B.append(img_B)

    train_A = np.asarray(train_img_A)
    train_B = np.asarray(train_img_B)
    print (train_A.shape)
    print (train_B.shape)

    test_all = glob.glob(maps_dir + "/val/*.jpg")
    test_img_A = []
    test_img_B = []

    for file in test_all:
        full_image = np.array(imread(file).resize((512, 256), Image.ANTIALIAS))
        img_B = full_image[:, full_image.shape[1] // 2:, :]
        img_A = full_image[:, :full_image.shape[1] // 2, :]
        test_img_A.append(img_A)
        test_img_B.append(img_B)

    test_A = np.asarray(test_img_A)
    test_B = np.asarray(test_img_B)
    print (test_A.shape)
    print (test_B.shape)

    return train_A, train_B, test_A, test_B


# train_all = glob.glob(maps_dir + "/train/*.jpg")
batch_size = 1


def load_batch_image(idx, maps_dir):
    train_all = glob.glob(maps_dir + "/train/*.jpg")
    full_image = np.array(imread(train_all[idx]).resize((512,256), Image.ANTIALIAS))
    img_B = full_image[:, full_image.shape[1] // 2:, :]
    img_A = full_image[:, :full_image.shape[1] // 2, :]

    return img_A, img_B


# test_all = glob.glob("maps/val/*.jpg")


def load_test_image(idx, maps_dir):
    test_all = glob.glob(maps_dir + "/val/*.jpg")
    full_image = np.array(imread(test_all[idx]).resize((512, 256), Image.ANTIALIAS))
    img_A = full_image[:, :full_image.shape[1] // 2, :] / 255.

    return img_A


def save_images(image, size, img_path):
    return imsave(image, size, img_path)


def imsave(image, img_size, img_path):
    # image = Image.fromarray(np.squeeze(image * 255.).astype(np.uint8))
    return Image.fromarray(np.squeeze(image * 255.).astype(np.uint8)).save(img_path)
    # return mp.imsave(np.squeeze(img_path), image)


def inverse_transform(image):
    return (image + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('In merge function, the first argument must have dimensions: HxW or HxWx3 or HxWx4')
