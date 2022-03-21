#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import pickle as pk
from tqdm import tqdm
import numpy as np

import logging
import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y.%m.%d. %H:%M:%S',
                    )
log = logging.getLogger(__name__)


def is_dir_exist(_dir, remove_before=False):
    if os.path.exists(_dir):
        if remove_before:
            # os.remove(_dir)
            os.rename(_dir, _dir + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            os.makedirs(_dir)
    else:
        log.info("Dir：[%s] created successfully!" % _dir)
        os.makedirs(_dir)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pk.load(f, encoding='bytes')
    return dict


def get_per_pixel_mean(all_imgs):
    arr = np.array(all_imgs, dtype='float32')
    mean = np.mean(arr, axis=0)
    return mean


def get_per_channel_mean(all_imgs):
    mean = get_per_pixel_mean(all_imgs)
    return np.mean(mean, axis=(0, 1))


def read_cifar(dataset_dir):
    cifar = unpickle(dataset_dir)
    cifar_data = cifar[b'data']
    cifar_label = cifar[b'labels']
    image_count = len(cifar_data)
    imgs = []
    for i in range(image_count):
        img = cifar_data[i].reshape(3, 32, 32)
        img = np.transpose(img, [1, 2, 0])
        imgs.append(img)
    return imgs, cifar_label


def image_preprocess(image_set):
    print("start preprocessing dataset!")
    mean = get_per_pixel_mean(image_set)
    # seed = random.randint(0, 10)
    result_image_set = []
    for img in tqdm(image_set):
        img = img - mean
        result_image_set.append(img)
    return result_image_set


def load_original_test_dataset(cifar_dir):
    test_data_file = os.path.join(cifar_dir, 'test_batch')
    cifar_data, cifar_label = read_cifar(test_data_file)
    return cifar_data, cifar_label


def save_test_batch_bin(cifar_dir, batch_size, bin_dir, label_dir):
    image_set, label_set = load_original_test_dataset(cifar_dir)
    image_count = len(image_set)
    if image_count % batch_size != 0:
        extend_num = ((image_count // batch_size) + 1) * batch_size - image_count
        print(extend_num)
        image_set = np.concatenate((image_set, image_set[:extend_num]), axis=0)
        label_set = label_set + label_set[:extend_num]
    batch_index = 1
    for i in tqdm(range(0, len(image_set), batch_size)):
        test_batch_data = image_set[i: i + batch_size]
        test_batch_label = label_set[i: i + batch_size]

        test_batch_data_filename = '%d_batch_%d_%d.bin' % (batch_index, i, i + batch_size)
        test_batch_data_dir = os.path.join(bin_dir, test_batch_data_filename)

        test_batch_label_filename = '%d_batch_%d_%d.txt' % (batch_index, i, i + batch_size)
        test_batch_label_dir = os.path.join(label_dir, test_batch_label_filename)

        if isinstance(test_batch_data, list):
            test_batch_data = np.array(test_batch_data[0])
            test_batch_label = np.array(test_batch_label[0])

        with open(test_batch_label_dir, 'w+') as f:
            for num in range(len(test_batch_label)):
                f.write("%d" % test_batch_label[num])
            f.write('\n')

        test_batch_data = np.array(test_batch_data, dtype=np.float32)
        test_batch_data.tofile(test_batch_data_dir)
        batch_index += 1


if __name__ == "__main__":
    cifar_dir = "./cifar-10-batches-py/"
    batch_size = 64
    bin_dir = "./test_bin_batchSize_%d" % batch_size
    label_dir = "./test_label_batchSize_%d" % batch_size
    is_dir_exist(bin_dir)
    is_dir_exist(label_dir)
    save_test_batch_bin(cifar_dir, batch_size, bin_dir, label_dir)
