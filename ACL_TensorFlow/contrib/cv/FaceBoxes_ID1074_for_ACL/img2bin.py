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
import numpy as np
import json
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
import math
import io
import os
import tensorflow as tf
import PIL.Image
import numpy as np

faceList_path=r"../FDDBFile/faceList.txt"
originalPics=r"../FDDB/originalPics"
Bin_DIR=r"../binfiles/"


if not os.access(Bin_DIR,os.F_OK):
    os.mkdir(Bin_DIR)

pic_name=[]
with open(faceList_path) as f:
    temp=f.readlines()
    for i in temp:
        pic_name.append(i.strip())


num_examples = len(pic_name)
print('Number of images:', num_examples)


for example in tqdm(pic_name):

    image_path = originalPics+"/"+example+".jpg"
    image_name = example.replace("/","-")+".jpg"
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    image_=np.asarray(image)
    image__=cv2.resize(image_,(1024, 1024), interpolation=cv2.INTER_CUBIC)
    image__.tofile(Bin_DIR+"{}.bin".format(image_name))


print('Result is here:', Bin_DIR)