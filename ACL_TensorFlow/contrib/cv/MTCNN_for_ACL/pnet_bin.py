# Copyright 2020 Huawei Technologies Co., Ltd
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

from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import sys,os
import numpy as np
import six
import math
import cv2
import subprocess

min_face=20

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def processed_image(img, scale):
    '''预处理数据，转化图像尺度并对像素归一到[-1,1]
    '''
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    img_resized = (img_resized - 127.5) / 128
    return img_resized

def transbin(data_in_om,test_dir):

    path = test_dir
    indx = 0
    for item in os.listdir(path):

        img_path = os.path.join(path, item)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        net_size = 12
        # 人脸和输入图像的比率
        current_scale = float(net_size) / min_face
        im_resized = processed_image(img, current_scale)
        current_height, current_width, _ = im_resized.shape

        # 类别和box
        image_reshape = np.reshape(im_resized, [1, current_height, current_width, 3])
        print("height", current_height)
        print("width", current_width)
        image_reshape = image_reshape.astype(np.float32)

        os.system("mkdir {}".format(data_in_om))
        image_reshape.tofile(os.path.join("{}".format(data_in_om), "{}_pnet_input.bin".format(indx)))

        indx = indx + 1

if __name__ == '__main__':
    transbin(sys.argv[1],sys.argv[2])



