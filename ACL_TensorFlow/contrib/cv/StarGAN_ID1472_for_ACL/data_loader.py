# -*- coding:utf-8 -*-
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
Load image and attribute label with numpy.
Refer to data/data_loader.py, first generate test_filenames, test_labels; train_filenames, train_labels. then,
use opencv to load image.
"""
import os
import cv2
import random
import numpy as np


def save_images(image, image_path, max_samples=16):
    # batch_size is 16.
    image = image[0:max_samples, :, :, :]
    image = np.concatenate([image[i, :, :, :] for i in range(max_samples)], axis=0)

    image = (((image + 1.)/2).clip(0., 1.))*255

    # save image
    cv2.imwrite(image_path, np.uint8(image))


# Load image
def _load_image(image_name, image_root, target_height=128, target_width=128, channels=3, as_float=True, is_training=False):
    # ues skimage to load image, return numpy ndarray. the ndarray format is RGB, shape is (C, H, W).
    if channels == 3:
        mode = 1
    else:
        mode = 0
    image = cv2.imread(os.path.join(image_root, image_name), mode)
    '''
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    '''
    if is_training:
        # RandomHorizontalFlip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # CenterCrop

        # Resize
        image = cv2.resize(image, (target_height, target_width))
        # ToTensor Normalize
        image = image.astype(np.float32)/127.5 - 1.

    else:
        image = cv2.resize(image, (target_height, target_width))
        image = image.astype(np.float32)/127.5 - 1.

    return image

# Load data, refer to StarGAN/data_loader.py. 
# Use a class CelebADataset.

class CelebADataset(object):
    def __init__(self, image_root, metadata_path, is_training, batch_size=None, image_h=None, image_w=None, image_c=None):
        # image_root: image root path, e.g. ./datasets/CelebA_nocrop/images.
        # metadata_path is the datasets/list_attr_celeba.txt, from 3rd line, the format is:
        # image_name attribute_labels
        self.image_root = image_root
        self.metadata_path = metadata_path
        self.is_training = is_training
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

        self.lines = open(self.metadata_path, 'r').readlines()
        # Use .readlines() return a list, the element oflist is the every line of datasets/list_attr_celeba.txt.
        # The end of line is \r\n. use strip() or split() to get rid of it.
        self.attr2idx = {}
        self.idx2attr = {}
        # dict. key is index, value is attribute label. 
        # attribute label is the 2nd line of datasets/list_attr_celeba.txt.

    def batch_generator_numpy(self):
        attrs = self.lines[1].split()
        # 2nd line, represent the attribute labels.
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
            # dict. key is index, value is attribute label.

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] # ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Male', 'Smiling', 'Wearing_Hat', 'Young']
        # selected_attrs, select the attributes. You can select attributes what you want.
        # list.
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        image_attr_lines = self.lines[2:]
        # from 3rd. every line is: image_name attribute_labels
        random.seed(1234)
        random.shuffle(image_attr_lines) # random shuffling. random can work on list directly.
        # numpy.random.shuffle() shuffle array.

        for i, line in enumerate(image_attr_lines):
            # from 3rd. every line is: image_name attribute_labels
            splits = line.split()
            # split(), defaut is ' '. image_name attribute_labels.
            filename = splits[0]
            # image path
            values = splits[1:]
            # attribute label.

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            # So, every image's attribute label is a list, its length is len(self.selected_attrs), w.t. 8.
            # the element of list is 1/0, represent this image has the attribute or not.

            # generate test_filenames, test_labels; train_filenames, train_labels.
            if (i + 1) < 2001:
                # test dataset capacity.
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

        # Read each JPEG file. 
        data_gen = {} # return a dict. key is string, value is numpy array.

        if self.is_training:
            # training
            print("Loading training datasets...")
            train_list_length = len(self.train_filenames)
            # total train iter
            if train_list_length % self.batch_size == 0:
                iter_until = train_list_length + self.batch_size
            else:
                iter_until = (train_list_length // self.batch_size + 1) * self.batch_size

            # load image and attribute labels.
            while True:
                for start, end in zip(range(0, iter_until, self.batch_size), \
                            range(self.batch_size, iter_until, self.batch_size)):

                    batch_images = self.train_filenames[start:end]
                    batch_labels = self.train_labels[start:end]

                    data_batch_images = np.array(list(map(lambda x: _load_image(x, image_root=self.image_root,
                        target_height=self.image_h, target_width=self.image_w, channels=self.image_c, is_training=self.is_training), batch_images)))
                    data_gen["images"] = data_batch_images
                    data_batch_labels = np.array(batch_labels)
                    # transform to numpy array directly.
                    data_gen["attribute"] = data_batch_labels
                    # test
                    # print(data_batch_images.shape)
                    # print(data_batch_labels.shape)
                    yield data_gen
        else:
            # testing
            print("Loading testing datasets...")
            test_list_length = len(self.test_filenames)
            # total test iter
            if test_list_length % self.batch_size == 0:
                iter_until = test_list_length + self.batch_size
            else:
                iter_until = (test_list_length // self.batch_size + 1) * self.batch_size

            # load image and attribute labels.
            while True:
                for start, end in zip(range(0, iter_until, self.batch_size), \
                            range(self.batch_size, iter_until, self.batch_size)):

                    batch_images = self.test_filenames[start:end]
                    batch_labels = self.test_labels[start:end]

                    data_batch_images = np.array(list(map(lambda x: _load_image(x, image_root=self.image_root,
                        target_height=self.image_h, target_width=self.image_w, channels=self.image_c), batch_images)))
                    data_gen["images"] = data_batch_images
                    data_batch_labels = np.array(batch_labels)
                    # transform to numpy array directly.
                    data_gen["attribute"] = data_batch_labels

                    # test
                    # print(data_batch_images.shape)
                    # print(data_batch_labels.shape)
                    yield data_gen

# test
if __name__ == '__main__':
    is_training = False
    data_loader = CelebADataset(image_root="./data/celeba/images",
                                metadata_path="./data/celeba/list_attr_celeba.txt",
                                is_training=is_training,
                                batch_size=8,
                                image_h=128, image_w=128, image_c=3)
    data_loader.batch_generator_numpy()
