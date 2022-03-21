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


import cfg
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", default="../image_test1", help="path of original pictures")
    parser.add_argument("--dst_path", default="../image_test_bin", help="path of output bin files")
    parser.add_argument("--pic_num", default=10000, help="picture number")
    args = parser.parse_args()
    src_path = args.src_path
    dst_path = args.dst_path
    pic_num  = args.pic_num
    files = os.listdir(src_path)
    files.sort()
    n = 0
    for file in files:
        src = src_path + "/" + file
        print("start to process %s"%src)
        img = image.load_img(src)
        d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        img = image.img_to_array(img,dtype=np.float32)
        print(img.shape)
        print(d_wight, d_height)
        if d_height != 736:
            zero = np.zeros((736 - d_height, d_wight, 3),dtype=np.float32)
            img = np.concatenate((img, zero), axis=0)
        print(img.shape)
        if d_wight != 736:
            zero = np.zeros((736, 736 - d_wight, 3),dtype=np.float32)
            img = np.concatenate((img, zero), axis=1)
        print('img.shape', img.shape)
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)
        print('x.shape', x.shape)
        print(x.dtype)
        x.tofile(dst_path + "/" + file + ".bin")
        n += 1
        if int(pic_num) == n:
            break