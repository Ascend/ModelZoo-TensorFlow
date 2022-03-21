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
"""  transform .bin file to .jpg picture.

python3 bin2jpg.py --in_XtoY "/home/ma-user/modelarts/inputs/data_url_0/bin/avatar2real_20220118_164237"
--in_YtoX '/home/ma-user/modelarts/inputs/data_url_0/bin/real2avatar_20220118_163317'
--out_XtoY '/home/ma-user/modelarts/outputs/train_url_0/jpg_avatar2real'
--out_YtoX '/home/ma-user/modelarts/outputs/train_url_0/jpg_real2avatar'
"""

import numpy as np
from PIL import Image
import os
import tensorflow as tf
from utils import save_images
import imageio
from skimage import img_as_ubyte
import argparse


parser = argparse.ArgumentParser(description='')
# 输入结果bin：/home/ma-user/modelarts/inputs/data_url_0/bin/avatar2real_20220118_164237
# /home/ma-user/modelarts/inputs/data_url_0/bin/real2avatar_20220118_163317
parser.add_argument('--in_XtoY', type=str,
                    default='/home/ma-user/modelarts/inputs/data_url_0/bin/avatar2real_20220118_164237',
                    help = 'bin for output real image.')
parser.add_argument('--in_YtoX', type=str,
                    default='/home/ma-user/modelarts/inputs/data_url_0/bin/real2avatar_20220118_163317',
                    help='bin for output avatar image.')

# 输出结果jpg：/home/ma-user/modelarts/outputs/train_url_0/jpg_avatar2real
# /home/ma-user/modelarts/outputs/train_url_0/jpg_real2avatar
parser.add_argument('--out_XtoY', type=str,
                    default='/home/ma-user/modelarts/outputs/train_url_0/jpg_avatar2real',
                    help='jpg for output real image.')
parser.add_argument('--out_YtoX', type=str,
                    default='/home/ma-user/modelarts/outputs/train_url_0/jpg_real2avatar',
                    help='jpg for output avatar image.')

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.out_XtoY):
    os.makedirs(FLAGS.out_XtoY)
if not os.path.exists(FLAGS.out_YtoX):
    os.makedirs(FLAGS.out_YtoX)


def parse_dir(input_dir, output_dir):
    num = 1
    for file in os.listdir(input_dir):
        if file.endswith('.bin'):
            data_dir = input_dir + "/" + file
            print("start to process %s" % data_dir)
            data = np.fromfile(data_dir, dtype=np.float32)
            # "uint8": 256*256*4
            # np.int32: 256*256 | I 32位整型像素
            # np.float32: 256*256 | F 32位浮点型像素

            # data = data.reshape(256, 256, 1)
            # im = Image.fromarray(data)
            # im.convert('RGB').save(output_dir + "/" + "out" + str(num) + ".jpg")  # 输出全黑

            data = data.reshape(256, 256)
            im = (data + 1.) * 127.5
            # print(np.min(im), np.max(im))  # -0.99365234 1.0009766
            # save_images(images=im, size=[1, 1],
            #             image_path=output_dir + "/" + "out" + str(num) + ".jpg")
            imageio.imwrite(output_dir + "/" + "out" + str(num) + ".jpg", im)
            num = num + 1


def main(_):
    print('Transfer output for XtoY model...')
    parse_dir(FLAGS.in_XtoY, FLAGS.out_XtoY)
    print('Transfer output for YtoX model...')
    parse_dir(FLAGS.in_YtoX, FLAGS.out_YtoX)


if __name__ == '__main__':
    tf.app.run()





