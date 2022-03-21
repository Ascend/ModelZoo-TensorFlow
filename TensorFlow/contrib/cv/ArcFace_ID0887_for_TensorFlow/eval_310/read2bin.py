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


import io
import pickle
import numpy as np
import imageio
from PIL import Image
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data',
                        default='/home/HwHiAiUser/arcface/dataset')
    parser.add_argument('--save_path', type=str, help='path to save TFRecord file',
                        default='/home/HwHiAiUser/arcface/data_bin')

    return parser.parse_args()


def generate_bin(path, image_size, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    # print(num)
    # print(len(issame_list))
    # print(issame_list)
    # images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float)
    # images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float)
    # m = config['augment_margin']
    # s = int(m/2)
    cnt = 0
    for bin in bins:
        # img = misc.imread(io.BytesIO(bin))
        img = imageio.imread(io.BytesIO(bin))
        # img = misc.imresize(img, [image_size, image_size])
        img = np.array(Image.fromarray(img).resize((image_size, image_size)), dtype=np.float32)
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img / 127.5 - 1.0
        img_f = img_f / 127.5 - 1.0
        img.tofile("%s/data_%d.bin" % (dst_path, cnt))
        img_f.tofile("%s/data_f_%d.bin" % (dst_path, cnt))
        cnt += 1
        if (cnt % 100 == 0):
            print(cnt)
    print('done!')


if __name__ == '__main__':
    args = get_args()
    val_data = {'agedb_30': 'agedb_30.bin',
                'lfw': 'lfw.bin',
                'cfp_ff': 'cfp_ff.bin',
                'cfp_fp': 'cfp_fp.bin',
                'calfw': 'calfw.bin',
                'cplfw': 'cplfw.bin',
                'vgg2_fp': 'vgg2_fp.bin'}
    print("begin convert")
    for k, v in val_data.items():
        generate_bin(os.path.join(args.read_dir, v), args.image_size, os.path.join(args.save_path, k))
        print('%s finish' % k)
