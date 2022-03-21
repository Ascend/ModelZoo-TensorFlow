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
import os
import tensorflow as tf
import cv2

if __name__ == '__main__':
    image_h = 256
    image_w = 256
    image_c = 1
    # data dir.
    data_BSD = "../datasets/BSDS_Quality100/train"
    data_BSD_Quality10 = "../datasets/BSDS_Quality10/train"
    #os.mkdir(os.path.join("../datasets", "tfrecords", "VOC0712_v2.tfrecords"))
    # TFRecordWriter, dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", "BSD.tfrecords"))
    #writer = tf.compat.v1.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", "VOC0712_v2.tfrecords"))
    with open("../datasets/BSDS300/iids_train.txt", "r") as fo:
        # image_files = fo.readlines() # return list
        for line in fo:
            line = line.strip() # 去掉\n, 空格! Necessary! String.
            # line = line.split() # not need
            # line[2:], e.g. is 'VOC2012/2010_005111.jpg'
            image_BSD = str(os.path.join(data_BSD, line[0:]))+".jpg"
            image_BSD_Quality10 = str(os.path.join(data_BSD_Quality10, line[0:]))+".jpg"
            print(image_BSD)
            print(image_BSD_Quality10)

            # Load image.
            image_clean = tf.gfile.FastGFile(image_BSD, 'rb').read()
            # image data type is string. read and binary.
            image_noisy = tf.gfile.FastGFile(image_BSD_Quality10, 'rb').read()


            # bytes write to Example proto buffer.
            example = tf.train.Example(features=tf.train.Features(feature={
                "image_clean": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_clean])),
                "image_noisy": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_noisy]))
                }))

            writer.write(example.SerializeToString()) # serialize to string.

    fo.close()
    writer.close()

