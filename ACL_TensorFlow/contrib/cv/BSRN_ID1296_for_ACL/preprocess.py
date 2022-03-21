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

"""  transform .jpg picture to .bin format.

"""
import os
import argparse

import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
scale_path = '/mnt/data1/2021/hht/huawei/bsrn/temp/bin/scale'
# 将图像数据转换为bin文件, images大小固定为120*80
def tobin(imgdir,bindir):
    # images = []
    for file in os.listdir(imgdir):
        if file.endswith('.png'):
            pic_path = imgdir + "/" + file
            print("start to process %s" % pic_path)

            image = tf.read_file(filename=pic_path)
            image = tf.image.decode_png(image, channels=3,dtype=tf.uint8)
            print("image's shape: ",image.shape)
            image = tf.image.resize_images(image, size=(480, 320))
            # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # image.set_shape([256, 256, 3])
            # print(image.shape)

            with tf.Session() as sess:
                image_numpy = image.eval()
                # print(image_numpy.shape)
            # save the pic as .bin format for Ascend310 infer.
            image_numpy = np.array(image_numpy.reshape(-1),dtype=np.float32)
            image_numpy.tofile(bindir + "/" + file + ".data.bin")
            # scale = np.array(4,dtype=np.uint8)
            # scale.tofile(scale_path + "/" + file + ".data.bin")
            # images.append(image_numpy)
    # images = np.array(images,dtype=np.uint8)
    # print("----------------------------------------------------------")
    # print("images type:", type(images), "shape: ", images.shape)
    # images.tofile(bindir + "/" + "data.bin")

if __name__ == '__main__':
    # argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", type=str, default='', help="Input dataset path.")
    parser.add_argument("--data_truth_dir", type=str, default='', help="Input label dataset path.")
    parser.add_argument("--bin_input_dir", type=str, default='', help="Output dataset path.")
    parser.add_argument("--bin_truth_dir", type=str, default='', help="Output label dataset path.")
    config = parser.parse_args()
    # preparation
    if os.path.exists(config.bin_input_dir + '/x2'):
        pass
    else:
        os.makedirs(config.bin_input_dir + '/x2')
    if os.path.exists(config.bin_input_dir + '/x3'):
        pass
    else:
        os.makedirs(config.bin_input_dir + '/x3')
    if os.path.exists(config.bin_input_dir + '/x4'):
        pass
    else:
        os.makedirs(config.bin_input_dir + '/x4')
    if os.path.exists(config.bin_truth_dir):
        pass
    else:
        os.makedirs(config.bin_truth_dir)

    # tobin(config.data_input_dir + '/x2', config.bin_input_dir + '/x2')
    # tobin(config.data_input_dir + '/x3', config.bin_input_dir + '/x3')
    # tobin(config.data_input_dir + '/x4', config.bin_input_dir + '/x4')
    tobin(config.data_truth_dir, config.bin_truth_dir)