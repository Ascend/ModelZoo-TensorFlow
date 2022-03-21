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
# -*- coding:utf-8 -*-
"""
Generate TFRecords file, for training.
1. Generate the list file, include VOC2007 and VOC2012 dataset. The list file format is:
path/image file.
"""
from npu_bridge.npu_init import *
import os
import tensorflow as tf

if __name__ == '__main__':
    # data dir.

    data_VOC0712 = "../datasets/VOC0712"
    data_VOC0712_Quality10 = "../datasets/VOC0712_Quality10"

    # TFRecordWriter, dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", "VOC0712.tfrecords"))

    with open("../datasets/VOC0712.txt", "r") as fo:
        # image_files = fo.readlines() # return list
        for line in fo:
            line = line.strip() # 去掉\n, 空格! Necessary! String.
            # line = line.split() # not need
            # line[0:], e.g. is 'VOC2012/2010_005111.jpg'
            image_VOC0712 = str(os.path.join(data_VOC0712, line[0:]))
            image_VOC0712_Quality10 = str(os.path.join(data_VOC0712_Quality10, line[0:]))
            print(image_VOC0712)
            print(image_VOC0712_Quality10)

            # Load image.
            image_clean = tf.gfile.FastGFile(image_VOC0712, 'rb').read()  
            # image data type is string. read and binary.
            image_noisy = tf.gfile.FastGFile(image_VOC0712_Quality10, 'rb').read()  

            # bytes write to Example proto buffer.
            example = tf.train.Example(features=tf.train.Features(feature={
                "image_clean": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_clean])),
                "image_noisy": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_noisy]))
                }))

            writer.write(example.SerializeToString()) # serialize to string.

    fo.close()
    writer.close()

