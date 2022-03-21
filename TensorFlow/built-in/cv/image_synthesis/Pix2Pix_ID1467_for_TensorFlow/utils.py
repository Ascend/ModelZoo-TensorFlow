# -*- coding: utf-8 -*-
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
"""
Created on Sun Feb 24 21:44:04 2019

@author: wmy
"""
from npu_bridge.npu_init import *

import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

class DataLoader(object):
    
    def __init__(self, dataset_path=r'./datasets/CombinedImages'):
        self.image_height = 256
        self.image_width = 256
        self.dataset_path = dataset_path
        pass
    
    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    def find_images(self, path):
        result = []
        for filename in os.listdir(path):
            _, ext = os.path.splitext(filename.lower())
            if ext == ".jpg" or ext == ".png":
                result.append(os.path.join(path, filename))
                pass
            pass
        result.sort()
        return result
    
    def load_data(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        # 去除随机值
        # batch_images = np.array(search_result[:batch_size])
        batch_images = np.random.choice(search_result, size=batch_size)
        images_A = []
        images_B = []
        for image_path in batch_images:
            combined_image = self.imread(image_path)
            h, w, c = combined_image.shape
            nW = int(w/2)
            image_A, image_B = combined_image[:, :nW, :], combined_image[:, nW:, :]
            image_A = scipy.misc.imresize(image_A, (self.image_height, self.image_width))
            image_B = scipy.misc.imresize(image_B, (self.image_height, self.image_width))
            if not for_testing and np.random.random() < 0.5:
                # 数据增强，左右翻转
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)
                pass
            images_A.append(image_A)
            images_B.append(image_B)
            pass
        images_A = np.array(images_A)/127.5 - 1.
        images_B = np.array(images_B)/127.5 - 1.
        return images_A, images_B

    def process_function(self, image_path):
        def read_data(image_path):
            if isinstance(image_path, bytes):
                image_path = image_path.decode()
            combined_image = self.imread(image_path)
            h, w, c = combined_image.shape
            nW = int(w / 2)
            image_A = combined_image[:, :nW, :]
            image_B = combined_image[:, nW:, :]
            image_A = scipy.misc.imresize(image_A, (self.image_height, self.image_width))
            image_B = scipy.misc.imresize(image_B, (self.image_height, self.image_width))
            if np.random.random() > 0.5:
                # 数据增强，左右翻转
                image_A = np.fliplr(image_A)
                image_B = np.fliplr(image_B)

            image_A = (image_A/127.5 - 1.).astype(np.float32)
            image_B = (image_B/127.5 - 1.).astype(np.float32)

            return image_A, image_B

        return tf.py_func(read_data, inp=[image_path], Tout=[tf.float32, tf.float32])

    def make_dataset(self, batch_size, epoch):
        search_result = self.find_images(self.dataset_path)
        self.n_complete_batches = int(len(search_result) / batch_size)
        ds = tf.data.Dataset.from_tensor_slices(search_result)
        ds = ds.map(lambda imagepath: self.process_function(imagepath), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # same with data size for perfect shuffle
        # ds = ds.shuffle(buffer_size=400)
        ds = ds.repeat(epoch)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        return ds

    def load_batch(self, batch_size=1, for_testing=False):
        search_result = self.find_images(self.dataset_path)
        self.n_complete_batches = int(len(search_result) / batch_size)
        for i in range(self.n_complete_batches):
            batch = search_result[i*batch_size:(i+1)*batch_size]
            images_A, images_B = [], []
            for image_path in batch:
                combined_image = self.imread(image_path)
                h, w, c = combined_image.shape
                nW = int(w/2)
                image_A = combined_image[:, :nW, :]
                image_B = combined_image[:, nW:, :]
                image_A = scipy.misc.imresize(image_A, (self.image_height, self.image_width))
                image_B = scipy.misc.imresize(image_B, (self.image_height, self.image_width))
                if not for_testing and np.random.random() > 0.5:
                    # 数据增强，左右翻转
                    image_A = np.fliplr(image_A)
                    image_B = np.fliplr(image_B)
                    pass
                images_A.append(image_A)
                images_B.append(image_B)
                pass
            images_A = np.array(images_A)/127.5 - 1.
            images_B = np.array(images_B)/127.5 - 1.
            yield images_A, images_B
