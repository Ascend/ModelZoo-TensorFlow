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

import sys
import numpy as np
import cv2
import os
import argparse
from data_process.config import input_shape
import tensorflow as tf
from time import *


def get_bin_info(bin_path, info_name,imglist):
    # bin_list = os.listdir(file_path)
    # bin_list.sort(key=lambda x: int(x[:-4]))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    with open(info_name, 'w') as file:
        for index in range(len(imglist)):
            content = ' '.join([str(index), bin_path+"/"+str(imglist[index])+".bin",str(416),str(416)])
            print("content",content)
            file.write(content)
            file.write('\n')

def keras_pre(bin_path):
    total = 0
    current = 0
    imglist = []
    with open("data_process/2007_test.txt", "r") as f:
        for tmp in f.readlines():
            total += 1
            starttime = time()
            test_path = tmp.split(" ")[0]
            image = cv2.imread(test_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，转换为RGB
            image = cv2.resize(image, input_shape)
            image = image / 255
            image = np.expand_dims(image, 0).astype(np.float32)
            endtime = time()
            print("use time ",endtime-starttime)
            current += (endtime-starttime)
            print(image.dtype)
            test_path = test_path[-10:-4]
            print("test_path",test_path)
            print("image_size",image.shape)
            imglist.append(test_path)
            # print(bin_path+"/"+test_path+".bin")
            print(os.path.join(bin_path, test_path + ".bin"))
            image.tofile(os.path.join(bin_path, test_path + ".bin"))
    print("avg time",(current/total)*1000)
            # image.tofile(bin_path+"/"+test_path+".bin")
    # return imglist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of YAD2K model')
    parser.add_argument("--image_folder_path", dest="file_path",default="DataSet/VOCdevkit1/VOC2007/JPEGImages/", help='image of dataset')
    parser.add_argument("--bin_folder_path",dest="bin_path", default="./bin/demo_bin", help='Preprocessed image buffer')
    parser.add_argument("--bin-info_name", dest="info_name", default="./bin/demo_info", help='bin-->info')

    args = parser.parse_args()

    file_path = args.file_path  # DataSet/VOCdevkit1/VOC2007/JPEGImages/
    bin_path = args.bin_path  #./bin/demo_bin
    info_name = args.info_name  #./demo_info

    imglist = keras_pre(bin_path)
    # get_bin_info(bin_path,info_name,imglist)

