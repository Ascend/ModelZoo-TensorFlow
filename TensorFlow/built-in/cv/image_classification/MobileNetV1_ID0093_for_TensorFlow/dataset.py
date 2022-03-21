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
#!/usr/bin/env python
# -*- coding_utf-8 -*-

# Copy from https://blog.csdn.net/missyougoon/article/details/86549404


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
from imgaug import augmenters as iaa
import tensorflow as tf
import cv2
import os


class CifarData:
    def __init__( self, filenames, need_shuffle ):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = self.load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        self._data = np.vstack(all_data)

        self._data = self._data / 255.

        self._labels = np.hstack( all_labels )

        self._num_data = self._data.shape[0]

        self._need_shuffle = need_shuffle

        self._indicator = 0

        if self._need_shuffle:
            self._shffle_data()

    def load_data(self, filename):

        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            return data[b'data'], data[b'labels']

    def data_aug(self, img):
        seq = iaa.SomeOf((1,3), [
            iaa.Fliplr(1.0),
            iaa.GaussianBlur(0.5),
            iaa.Sharpen(alpha=0.5)
        ],random_order=True)
        return seq.augment_image(img)

    def _shffle_data( self ):
        p = np.random.permutation( self._num_data )
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch( self, batch_size):
        '''return batch_size example as a batch'''

        end_indictor = self._indicator + batch_size

        if end_indictor > self._num_data:
            if self._need_shuffle:

                self._shffle_data()

                self._indicator = 0

                end_indictor = batch_size

        batch_data = self._data[self._indicator:end_indictor]
        batch_labels = self._labels[self._indicator:end_indictor]
        self._indicator = end_indictor
        return batch_data, batch_labels

