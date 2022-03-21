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
# coding=utf-8
from .augement import Augmenter
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from .flowlib import read_flow
from .utils import crop_features

class Sintel:
    def __init__(self, batch_size, base_path, img_size, train_data=None, val_data=None, status='clean',augment=True):
        """
        :param batch_size: batch size
        :param num_classes: number of classes for datasets
        :param base_path: data set base path
        :param train_data: train file path
        :param val_data: validation file path
        :param status: type of sintel train set
        """
        self.batch_size = batch_size
        self.base_path = base_path
        self.val_data = val_data
        self.train_data = train_data
        self.image_size = img_size
        img_path = '{}/{}'.format(base_path, status)
        label_path = '{}/flow'.format(base_path)
        self.train = []
        self.val = []
        self.augement = Augmenter()
        self.augment = augment
        if self.train_data != None:
            with open(self.train_data, 'r') as f:
                imgs = f.readlines()
            self.trian_num = len(imgs)
            for i in imgs:
                i = i[:-1].split(',')

                image1 = Image.open(os.path.join(img_path,i[0]))
                image1 = np.array(image1)
                image1 = image1[..., [2, 1, 0]]
                image1 = image1 / 255.0
                # if image1.max() > 1.0:
                #     image1 = image1 / 255.0

                image2 = Image.open(os.path.join(img_path,i[1]))
                image2 = np.array(image2)
                image2 = image2[..., [2, 1, 0]]
                image2 = image2 / 255.0
                # if image2.max() > 1.0:
                #     image2 = image2 / 255.0

                label = read_flow(os.path.join(label_path,i[2]))
                self.train.append([image1, image2, label])
            print("Data Loading finished! train_num:{}".format(self.trian_num))
        if self.val_data != None:
            with open(self.val_data, 'r') as f:
                imgs = f.readlines()
            self.val_num = len(imgs)
            for i in imgs:
                i = i[:-1].split(',')

                image1 = Image.open(os.path.join(img_path,i[0]))
                image1 = np.array(image1)
                image1 = image1[..., [2, 1, 0]]
                image1 = image1 / 255.0
                # if image1.max() > 1.0:
                #     image1 = image1 / 255.0

                image2 = Image.open(os.path.join(img_path,i[1]))
                image2 = np.array(image2)
                image2 = image2[..., [2, 1, 0]]
                image2 = image2 / 255.0
                # if image2.max() > 1.0:
                #     image2 = image2 / 255.0

                label = read_flow(os.path.join(label_path,i[2]))
                self.val.append([image1, image2, label])

            print("Data Loading finished! val_num:{}".format( self.val_num))



    def get_batch(self, count):
        start = count * self.batch_size
        end = (count + 1) * self.batch_size
        start_pos = max(0, start)
        end_pos = min(end, self.trian_num)
        images_batch = self.train[start_pos: end_pos]
        images1 = []
        images2 = []
        labels = []
        for imgs in images_batch:

            image1 = imgs[0]
            image2 = imgs[1]
            label = imgs[2]
            
            if self.image_size is not None:
                h, w = image1.shape[:2]
                h_max, w_max = self.image_size
                h,w,h_max,w_max = int(h), int(w), int(h_max), int(w_max)
                assert (h >= h_max and w >= w_max)
                max_y_offset, max_x_offset = h - h_max, w - w_max
                if max_y_offset > 0 or max_x_offset > 0:
                    y_offset = np.random.randint(max_y_offset + 1)
                    x_offset = np.random.randint(max_x_offset + 1)
                    # The following assumes the image pair is in [2,H,W,3] format
                    image1 = image1[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                    image2 = image2[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                    label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]

            if self.augment:
                image, label = self.augement.augment([(image1, image2)], [label])
                image,label = image[0], label[0]
                images1.append(image[0])
                images2.append(image[1])
                labels.append(label)
            else:
                images1.append(image1)
                images2.append(image2)
                labels.append(label)
        datas1 = np.array(images1)
        datas2 = np.array(images2)
        labels = np.array(labels)

        return datas1, datas2, labels

    def get_train_batch_num(self):
        return self.trian_num // self.batch_size

    def get_val_batch_num(self):
        return self.val_num // self.batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_val_data(self,count):
        start = count * self.batch_size
        end = (count + 1) * self.batch_size
        start_pos = max(0, start)
        end_pos = min(end, self.val_num)
        images_batch = self.val[start_pos: end_pos]
        val_images1 = []
        val_images2 = []
        val_labels = []
        for imgs in images_batch:
            image1 = imgs[0]
            image2 = imgs[1]
            val_label = imgs[2]

            if self.image_size is not None:
                    h, w = image1.shape[:2]
                    h_max, w_max = self.image_size
                    h,w,h_max,w_max = int(h), int(w), int(h_max), int(w_max)
                    assert (h >= h_max and w >= w_max)
                    max_y_offset, max_x_offset = h - h_max, w - w_max
                    if max_y_offset > 0 or max_x_offset > 0:
                        y_offset = np.random.randint(max_y_offset + 1)
                        x_offset = np.random.randint(max_x_offset + 1)
                        # The following assumes the image pair is in [2,H,W,3] format
                        image1 = image1[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                        image2 = image2[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                        val_label = val_label[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
            val_images1.append(image1)
            val_images2.append(image2)
            val_labels.append(val_label)
        datas1 = np.array(val_images1)
        datas2 = np.array(val_images2)
        labels = np.array(val_labels)
        return datas1, datas2, labels

