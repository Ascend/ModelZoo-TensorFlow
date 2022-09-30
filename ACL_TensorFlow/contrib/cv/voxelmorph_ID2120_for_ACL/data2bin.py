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

# -*- coding: UTF-8 -*-
# python imports
import os
from argparse import ArgumentParser
import datagenerators
import numpy as np
import nibabel as nib
# npu imports
from npu_bridge.npu_init import *


def data2bin(data_path):
    # load and transfer atlas from provided files. The atlas we used is 160x192x224.
    atlas_vol = nib.load(os.path.join(data_path, 'atlas_abide_brain_crop.nii.gz')).dataobj[
        np.newaxis, ..., np.newaxis].astype('float32')
    atlas_vol.tofile(data_path + '/tgt.bin')

    # load and transfer the test data
    test_path = os.path.join(data_path, 'test/')
    seg_path = os.path.join(data_path, 'seg_affined/')

    # 获取当前路径下的文件名，返回List
    file_names = os.listdir(test_path)
    n_batches = len(file_names)

    for k in range(n_batches):
        vol_name = test_path + file_names[k]
        seg_name = seg_path + file_names[k].replace('brain', 'seg')
        # load subject test
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)
        X_vol = X_vol.astype('float32')

        # prepare model folder
        bin_dir = data_path+'test_bin'
        if not os.path.isdir(bin_dir):
            os.mkdir(bin_dir)
        X_vol.tofile(bin_dir+'/%03d.bin' % k)
        print(k)



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str,
                        dest="data_path", default='../Dataset-ABIDE/')

    args = parser.parse_args()
    print(args.data_path)
    data2bin(args.data_path)
