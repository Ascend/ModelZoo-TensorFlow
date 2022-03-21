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



# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import imageio
import os

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.float32)

def imsave(image, path):
    """
    image:输入图像
    path：路径
    """
    return imageio.imwrite(path, image)
  
  
def prepare_data(dataset):
    """
    dataset ： data
    """

    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    """
    x:输入图像
    """
    return tf.maximum(x, leak * x)


def input_setup(index):
    """
    index:索引
    """
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir = (imread(data_ir[index]) - 127.5)/127.5
    print(type(input_ir))
    input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h=input_ir.shape
    input_ir=input_ir.reshape([w, h, 1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h=input_vi.shape
    input_vi=input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    print(train_data_ir.shape)
    print(train_data_vi.shape)
    if i + 1 <= 9:
        train_data_ir.tofile("./tobin/Nato_camp/ir/" + '0'+ str(i+1)[0:2] + ".bin")
        train_data_vi.tofile("./tobin/Nato_camp/vi/" + '0'+ str(i+1)[0:2] + ".bin")
    else:
        train_data_ir.tofile("./tobin/Nato_camp/ir/" + str(i+1)[0:2] + ".bin")
        train_data_vi.tofile("./tobin/Nato_camp/vi/" + str(i+1)[0:2] + ".bin")
    print(type(train_data_vi[0, 1, 1, 0]))
    return train_data_ir, train_data_vi

foldername = 'Nato_camp'
data_ir=prepare_data('Test_img/'+foldername + '/ir')
data_vi=prepare_data('Test_img/'+foldername + '/vi') 
print(data_ir[0])
# ./msame --model "/home/HwHiAiUser/msame/colorization.om" --input "/home/HwHiAiUser/msame/data1,/home/HwHiAiUser/msame/data2" --output "/home/HwHiAiUser/msame/out/" --outfmt TXT

path = "./Test_img/Nato_camp/ir"
files = os.listdir(path)
i = 0
for file in files:
    print(file)
    if file.endswith('.bmp'):
        src = "./Test_img/else/ir/" + file
        print("start to process %s" % src)
        train_data_ir,train_data_vi = input_setup(i) # 对原始图片进行需要的预处理
    i += 1


