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
# implements of data loader who feeds the data to model, by tf.data

import npu_device
import tensorflow as tf
from .label_anchor import AnchorLabeler


class DataLoader(object):
    '''
    data pipeline from data_reader (image,label) to tf.data
    '''
    def __init__(self, data_reader, anchors, stride, img_size=640, anchor_assign_method='wh',
                 anchor_positive_augment=True):
        self.data_reader = data_reader
        self.anchor_label = AnchorLabeler(anchors,
                                          grids=img_size / stride,
                                          img_size=img_size,
                                          assign_method=anchor_assign_method,
                                          extend_offset=anchor_positive_augment)
        self.img_size = img_size

    def __call__(self, batch_size=8, anchor_label=True):
        dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([self.img_size, self.img_size, 3], [None, 5]))

        if anchor_label:  # when train
            dataset = dataset.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def transform(self, image, label):
        label_encoder = self.anchor_label.encode(label)
        return image, label_encoder
