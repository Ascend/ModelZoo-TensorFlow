#! /usr/bin/env python
# coding=utf-8
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
# @Author: Longxing Tan, tanlongxing888@163.com

import npu_device
import os
import cv2
import random
import numpy as np
from .augment_data import load_mosaic_image, random_perspective, random_flip, augment_hsv
from .image_utils import resize_image, xyxy2xywh, xywh2xyxy


class DataReader(object):
    '''
    read the image and label from the text information (generated by dataset/prepare_data.py)
    resize the image, and adjust the label rect if necessary
    augment the dataset (augment function is defined in dataset/augment_data.py)
    '''
    def __init__(self, annotations_dir, img_size=640, transforms=None, mosaic=False, augment=False, filter_idx=None, test=False):
        self.annotations_dir = annotations_dir
        self.annotations = self.load_annotations(annotations_dir)
        self.idx = range(len(self.annotations))
        self.img_size = img_size  # image_target_size
        self.transforms = transforms
        self.mosaic = mosaic
        self.augment = augment
        self.test = test
        self.images_dir = []
        self.labels_ori = []  # original labels

        if filter_idx is not None:  # filter some samples
            self.idx = [i for i in self.idx if i in filter_idx]
            print('filter {} from {}'.format(len(self.idx), len(self.annotations)))

        for i in self.idx:
            image_dir, label = self.parse_annotations(self.annotations[i])
            self.images_dir.append(image_dir)
            self.labels_ori.append(label)

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, idx):
        if self.test:
            img = self.load_image(idx)
            img = resize_image(img, self.img_size, keep_ratio=True)
            return img
        if self.mosaic:  # mosaic need to load 4 images
            mosaic_border = [-self.img_size // 2, -self.img_size // 2]
            img, label = load_mosaic_image(idx, mosaic_border, self.img_size, self.images_dir, self.labels_ori)
        else:
            img, label = self.load_image_and_label(idx)

        if self.transforms:
            img, label = self.transforms(img, label, mosaic=self.mosaic, augment=self.augment)
        img, label = resize_image(img, self.img_size, keep_ratio=True, label=label)  # resize the image
        label[:, 0:4] = xyxy2xywh(label[:, 0:4])  # transfer xyxy to xywh
        return img, label

    def iter(self):
        for i in self.idx:
            yield self[i]

    def load_annotations(self, annotations_dir):
        with open(annotations_dir, 'r') as f:
            annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        print('Load examples : {}'.format(len(annotations)))
        if 'train' in annotations_dir:
            np.random.shuffle(annotations)
        return annotations

    def parse_annotations(self, annotation):
        example = annotation.split()
        image_dir = example[0]

        label = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])
        # assert label.shape[1] == 5, "Label have and only have 5 dims: xmin, ymin, xmax, ymax, class"
        # assert np.max(label[:, 0:4]) <= 1, "Label box should be (0, 1), {}".format(annotation)
        return image_dir, label

    def load_image(self, idx):
        img_dir = self.images_dir[idx]
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return img

    def load_image_and_label(self, idx):
        img = self.load_image(idx)
        label = self.labels_ori[idx].copy()
        return img, label

    def load_mixup_image_and_label(self, idx):
        img, label = self.load_image_and_label(idx)
        r_img, r_label = self.load_image_and_label(random.randint(0, len(self.images_dir) - 1))
        return (img + r_img) / 2, np.vstack((label, r_label)).astype(np.int32)


def transforms(img, label, mosaic, augment):
    # it's also easy to use albumentations here
    if augment:
        if not mosaic:        
            img, label = random_perspective(img, label)
        img = augment_hsv(img)

    if augment:  # flip the data if it helps
        img, label = random_flip(img, label)

    img = img / 255.  # normalize the image
    if np.max(label[:, 0:4]) > 1:  # normalize the bbox
        label[:, [0, 2]] = label[:, [0, 2]] / img.shape[1]
        label[:, [1, 3]] = label[:, [1, 3]] / img.shape[0]
    return img, label
