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
import os
import numpy as np
import argparse

import imageio
from PIL import Image

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

def get_img(src, img_size=False):
   img = imageio.imread(src, pilmode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = np.array(Image.fromarray(img).resize(img_size[:2]))
   return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", default="./datasets/imagenet_10", help="path of original pictures")
    parser.add_argument("--dst_path", default="./input", help="path of output bin files")
    parser.add_argument("--pic_num", default=-1, help="picture number")
    args = parser.parse_args()
    src_path = args.src_path
    dst_path = args.dst_path
    pic_num  = args.pic_num
    content_targets = _get_files(src_path)
    batch_shape = (len(content_targets),) + (256, 256, 3)
    X_batch = np.zeros(batch_shape, dtype=np.float32)
    for j, img_p in enumerate(content_targets):
        X_batch[j] = get_img(img_p, (256,256,3))
        imageio.imsave(dst_path+"/" + str(j)+".bin", X_batch[j])
    print("done")