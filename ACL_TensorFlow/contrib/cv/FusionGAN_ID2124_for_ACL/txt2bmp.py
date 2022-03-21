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

import struct
import os
import cv2 as cv
import numpy as np
import imageio
import pandas as pd


def loadDatadet(infile):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    return dataset

def imsave(image, path):
  return imageio.imwrite(path, image)

if __name__ == '__main__':
    print("*************************")
    filepath='./result/20211209_103004'
    files = os.listdir(filepath)
    i = 0
    for file in files:
        print(file)
        if file.endswith('.txt'):
            infile=loadDatadet(filepath+"/"+file)
            data = np.array(infile)
            data = np.multiply(np.float32(data),np.float32(127.5))+(np.float32(127.5))
            data = data.reshape([270,360])
            imsave(data, "./result/epoch8/"+str(i+1)+".bmp")
        i+=1