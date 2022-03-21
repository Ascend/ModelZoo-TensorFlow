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
import numpy as np
import pickle as pk
from tqdm import tqdm


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pk.load(f, encoding='bytes')
    return dict


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


def load_original_test_dataset():
    cifar_dir = "./cifar-10-batches-py/"

    test_data_file = os.path.join(cifar_dir, 'test_batch')
    cifar_data, cifar_label = read_cifar(test_data_file)
    return cifar_data, cifar_label


def load_txt(file):
    result = []
    with open(file, "r") as f:  # 打开文件
        while True:
            data = f.readline()  # 读取文件
            if not data:
                return result

            data = data.split(' ')
            result_line = [float(i) for i in data[:-1]]
            result.append(result_line)


def load_label(file):
    with open(file, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        result = [int(i) for i in data[:-1]]
        return result


def calculate_top1_err(ypred, yture):
    flag = np.zeros(10)
    assert len(ypred) == len(yture)
    err_count = 0
    for i in tqdm(range(len(ypred))):
        y_ture, y_pred = yture[i], ypred[i]
        if y_ture != y_pred:
            flag[y_ture] += 1
            err_count += 1
    print(err_count, len(ypred))
    print(flag)
    return err_count / len(ypred) * 1.


if __name__ == '__main__':

    pred_dir = "./output/2021128_21_52_17_409742"
    label_dir = "./test_label_batchSize_64"

    label_set = load_original_test_dataset()

    pred_file_list = os.listdir(pred_dir)
    ypred_list = []
    for file in pred_file_list:
        if file.endswith(".txt"):
            ypred_batch = load_txt(pred_dir + '/' + file)
            for batch in range(len(ypred_batch)):
                ypred_list.append(np.argmax(ypred_batch[batch]))

    label_file_list = os.listdir(label_dir)
    yture_list = []
    for file in label_file_list:
        if file.endswith(".txt"):
            yture_batch = load_label(label_dir + '/' + file)
            yture_list.extend(yture_batch)

    top5_err = calculate_top1_err(ypred_list, yture_list)
    print("OM-Top1-err: %.4f" % top5_err)
    print("GPU-Top1-err: 0.050")
    print("NPU-Top1-err: 0.048")
