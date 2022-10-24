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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imlib as im
import numpy as np

import data
import pylib
import os


def bin2img(data_dir='/home/TestUser03/code/mixmatch/data/ML_DATA/SSL',
            bin_dir='/home/TestUser03/code/mixmatch/mix_model/out/20221010_16_56_16_859201',
            save_dir='/home/TestUser03/code/mixmatch/mix_model/out_img',
            batch_size=1,
            img_size=128, atts=None):
    if atts is None:
        atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
                'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    pylib.mkdir(save_dir)

    te_data = data.Celeba(data_dir, atts, img_size, batch_size, part='test')
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        for i in range(len(atts) + 1):
            img = np.fromfile(os.path.join(bin_dir, "{}_{}_output_0.bin".format(idx, i)), dtype='float32')
            img.shape = 1, 128, 128, 3
            x_sample_opt_list.append(img)
        sample = np.concatenate(x_sample_opt_list, 2)
        im.imwrite(sample.squeeze(0), '%s/%06d.png' % (save_dir, idx + 182638))
        print('%06d.png done!' % (idx + 182638))
        if idx + 182638 == 202599: break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/TestUser03/code/mixmatch/data/ML_DATA/SSL',
                        help='the path of celeba(.jpg)')
    parser.add_argument('--bin_dir', type=str,
                        default='/home/TestUser03/code/mixmatch/mix_model/out/20221010_16_56_16_859201',
                        help='the path of generator image(.bin)')
    parser.add_argument('--save_dir', type=str, default='/home/TestUser03/code/mixmatch/mix_model/out_img',
                        help='the path of generator image(.jpg)')
    parser.add_argument('--img_size', type=int, default=128, help='input image size')
    att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    parser.add_argument('--atts', default=att_default, choices=data.Celeba.att_dict.keys(), nargs='+',
                        help='Attributes to modify by the model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    data_dir = args.data_dir
    bin_dir = args.bin_dir
    save_dir = args.save_dir
    img_size = args.img_size
    atts = args.atts
    batch_size = args.batch_size

    bin2img(data_dir, bin_dir, save_dir, batch_size, img_size, atts)