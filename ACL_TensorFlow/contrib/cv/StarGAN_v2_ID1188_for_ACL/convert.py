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
from glob import glob
from tqdm import tqdm
import os
import argparse

from utils import *

def img2bin(dataset_name, img_width, img_height, img_ch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_files = glob('./dataset/{}/{}/*.jpg'.format(dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(dataset_name, 'test'))
    for sample_file in tqdm(test_files):
        sample_image = load_test_image(sample_file, img_width, img_height, img_ch)
        print(sample_image.shape)
        sample_image.tofile(os.path.join(save_dir, '{}'.format(os.path.basename(sample_file)).split('.')[0] + ".bin"))
    print("finished!")

def bin2img(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    bin_files = glob('./{}/*.bin'.format(src_dir))
    for bin_file in tqdm(bin_files):
        img = np.fromfile(bin_file, dtype='float32')
        img.shape = 1, 256, 256, 3
        print(os.path.basename(bin_file))
        cur_path = os.path.join(dst_dir, '{}'.format(os.path.basename(bin_file)).split('.')[0] + '.jpg')
        save_images(img, [1, 1], cur_path)

if __name__ == '__main__':
    desc = "StarGAN_v2 for ACL"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='img2bin', help='img2bin or bin2img')
    parser.add_argument('--dataset', type=str, default='afhq-raw', help='dataset')
    parser.add_argument('--height', type=int, default=256, help='height')
    parser.add_argument('--width', type=int, default=256, help='width')
    parser.add_argument('--channels', type=int, default=3, help='width')
    parser.add_argument('--bin_path', type=str, default='./inputs', help='path to store bin file')
    parser.add_argument('--src_dir', type=str, default='./output/20211128_191252/', help='path of bin file')
    parser.add_argument('--dst_dir', type=str, default='./generated', help='path to store the generated images')



    args = parse_args()
    if args.mode == 'img2bin':
        img2bin(args.dataset, args.height, args.width, args.channels, args.bin_path)
    elif args.mode == 'bin2img':
        bin2img(args.src_dir, args.dst_dir)
    else:
        print('wrong mode. check your config')
