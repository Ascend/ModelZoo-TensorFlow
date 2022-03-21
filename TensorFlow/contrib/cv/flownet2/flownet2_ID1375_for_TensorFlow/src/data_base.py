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
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.ops.lookup_ops import IdTableWithHashBuckets
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
        self.img_path = '{}/{}'.format(base_path, status)
        self.label_path = '{}/flow'.format(base_path)
        self.train1 = []
        self.train2 = []
        self.train3 = []
        self.val1 = []
        self.val2 = []
        self.val3 = []
        self.augement = Augmenter()
        self.augment = augment
        if self.train_data != None:
            with open(self.train_data, 'r') as f:
                self.train_imgs = f.readlines()
            self.trian_num = len(self.train_imgs)
            train_id = np.arange(self.trian_num)
            self.db_train = tf.data.Dataset.from_tensor_slices(train_id)
            self.db_train = self.db_train.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.trian_num, count=-1))
            self.db_train = self.db_train.map(lambda idx: tf.py_func(self.totensor, [idx], [tf.float32, tf.float32, tf.float32])).repeat().batch(self.batch_size)
            self.db_train = self.db_train.prefetch(buffer_size=self.batch_size )	
            print("Data Loading finished! train_num:{}".format(self.trian_num))
        if self.val_data != None:
            with open(self.val_data, 'r') as f:
                self.val_imgs = f.readlines()
            self.val_num = len(self.val_imgs)
            val_id = np.arange(self.val_num)
            self.db_test = tf.data.Dataset.from_tensor_slices(val_id)
            self.db_test = self.db_test.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.val_num, count=-1))
            self.db_test = self.db_test.map(lambda idx: tf.py_func(self.val_totensor, [idx], [tf.float32, tf.float32, tf.float32])).repeat().batch(self.batch_size)
            self.db_test = self.db_test.prefetch(buffer_size=self.batch_size )	

            print("Data Loading finished! val_num:{}".format( self.val_num))
    

    def totensor(self, id):
        # print(img)
        img = self.train_imgs[id]
        img = img[:-1].split(',')
        image1 = Image.open(os.path.join(self.img_path,img[0]))
        image1 = np.array(image1)
        image1 = image1[..., [2, 1, 0]]
        image1 = image1 / 255.0
        # if image1.max() > 1.0:
        #     image1 = image1 / 255.0

        image2 = Image.open(os.path.join(self.img_path,img[1]))
        image2 = np.array(image2)
        image2 = image2[..., [2, 1, 0]]
        image2 = image2 / 255.0
        # if image2.max() > 1.0:
        #     image2 = image2 / 255.0

        label = read_flow(os.path.join(self.label_path,img[2]))
        if self.image_size is not None:
            h, w = image1.shape[:2]
            h, w = int(h), int(w)
            h_max, w_max = self.image_size
            h_max, w_max = int(h_max), int(w_max)
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
            img1 = image[0]
            img2 = image[1]
        else:
            img1 = image1
            img2 = image2

        # img1 = tf.cast(img1, dtype=tf.float32)
        # img2 = tf.cast(img2, dtype=tf.float32)
        # label = tf.cast(label, dtype=tf.float32)
        return np.array(img1, dtype=np.float32),np.array(img2, dtype=np.float32), np.array(label, dtype=np.float32)
        # return img1, img2, label

    def val_totensor(self, id):
        # img = tf.cast(id, dtype=np.int)
        # print(img)
        img = self.val_imgs[id]
        img = img[:-1].split(',')
        image1 = Image.open(os.path.join(self.img_path,img[0]))
        image1 = np.array(image1)
        image1 = image1[..., [2, 1, 0]]
        image1 = image1 / 255.0
        # if image1.max() > 1.0:
        #     image1 = image1 / 255.0

        image2 = Image.open(os.path.join(self.img_path,img[1]))
        image2 = np.array(image2)
        image2 = image2[..., [2, 1, 0]]
        image2 = image2 / 255.0
        # if image2.max() > 1.0:
        #     image2 = image2 / 255.0

        label = read_flow(os.path.join(self.label_path,img[2]))
        if self.image_size is not None:
            h, w = image1.shape[:2]
            h, w = int(h), int(w)
            h_max, w_max = self.image_size
            h_max, w_max = int(h_max), int(w_max)
            assert (h >= h_max and w >= w_max)
            max_y_offset, max_x_offset = h - h_max, w - w_max
            if max_y_offset > 0 or max_x_offset > 0:
                y_offset = np.random.randint(max_y_offset + 1)
                x_offset = np.random.randint(max_x_offset + 1)
                # The following assumes the image pair is in [2,H,W,3] format
                image1 = image1[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                image2 = image2[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]
                label = label[y_offset:y_offset + h_max, x_offset:x_offset + w_max, :]

        img1 = image1
        img2 = image2

        # img1 = tf.cast(img1, dtype=tf.float32)
        # img2 = tf.cast(img2, dtype=tf.float32)
        # label = tf.cast(label, dtype=tf.float32)
        return np.array(img1, dtype=np.float32),np.array(img2, dtype=np.float32), np.array(label, dtype=np.float32)
        # return img1, img2, label

    def get_train_batch_num(self):
        return self.trian_num // self.batch_size

    def get_val_batch_num(self):
        return self.val_num // self.batch_size

    def get_batch_size(self):
        return self.batch_size
