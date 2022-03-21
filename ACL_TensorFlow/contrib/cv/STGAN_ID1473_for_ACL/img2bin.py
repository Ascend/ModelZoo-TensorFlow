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
import numpy as np

import data
import pylib
import os


def img2bin(data_dir='/home/test_user02/yuxh/celeba/', save_dir='/home/test_user02/yuxh/celeba_bin/', img_size=128,
            batch_size=1,
            atts=None):
    if atts is None:
        atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
                'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

    path = [save_dir,
            os.path.join(save_dir, "input1"),
            os.path.join(save_dir, "input2"),
            os.path.join(save_dir, "input3")]
    pylib.mkdir(path)

    te_data = data.Celeba(data_dir, atts, img_size, batch_size, part='test')
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        b_sample_ipt_list = [a_sample_ipt.copy()]
        for i in range(len(atts)):
            tmp = np.array(a_sample_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]  # inverse attribute
            tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
            b_sample_ipt_list.append(tmp)
        raw_a_sample_ipt = a_sample_ipt.copy()
        raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * 0.5
        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * 0.5
            if i > 0:
                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * 2

            xa_sample_ipt.tofile(os.path.join(save_dir, "input1", "{}_{}.bin".format(idx, i)))
            _b_sample_ipt.tofile(os.path.join(save_dir, "input2", "{}_{}.bin".format(idx, i)))
            raw_a_sample_ipt.tofile(os.path.join(save_dir, "input3", "{}_{}.bin".format(idx, i)))
        print('%06d.jpg done!' % (idx + 182638))
        if idx + 182638 == 202599: break

    print("img2bin finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/test_user02/yuxh/celeba/',
                        help='the path of celeba(.jpg)')
    parser.add_argument('--save_dir', type=str, default='/home/test_user02/yuxh/celeba_bin/',
                        help='the path of celeba(.bin)')
    parser.add_argument('--img_size', type=int, default=128, help='input image size')
    att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    parser.add_argument('--atts', default=att_default, choices=data.Celeba.att_dict.keys(), nargs='+',
                        help='Attributes to modify by the model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    img_size = args.img_size
    atts = args.atts
    batch_size = args.batch_size

    img2bin(data_dir, save_dir, img_size, batch_size, atts)