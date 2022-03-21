"""
h5
"""
# coding=utf-8
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
import h5py as h5
import cv2
import os
import utils.compressed_sensing as cs
import matplotlib.pyplot as plt


def creat_h5file(file_path, save_file_name):
    """
    build h5 file from images
    input:
    param path:data path
    param save_name: save name
    """
    data = []
    mask_list = []
    for filename in os.listdir(file_path):
        data.append(cv2.imread(os.path.join(file_path, filename), 0))
        # file=h5.File(os.path.join(path, filename), 'r')['im_ori'][:]
        # data.append(file)
    data = np.array(data)
    data_shape = data.shape
    # mask_4=cs.cartesian_mask(shape=[data_shape[1],data_shape[2]],acc=3,sample_n=8)
    # with h5.File('mask_acc3.hdf5','w') as f:
    #     f.create_dataset('mask',data=mask_4)
    with h5.File('mask_acc3.hdf5', 'r') as f1:
        mask_4 = f1['mask'][:]
    for j in range(data_shape[0]):
        mask_list.append(mask_4)
    mask_list = np.array(mask_list)

    with h5.File(save_file_name, 'w') as f2:
        f2.create_dataset('label', data=data)
        f2.create_dataset('mask', data=mask_list)


if __name__ == '__main__':

    # path='data/chest_train'
    # save_name='chest_train_acc3.hdf5'
    path = 'data/chest_test'
    save_name = 'chest_test_acc4.hdf5'
    creat_h5file(path, save_name)
    with h5.File(save_name, 'r') as f:
        label = f['label'][:]
        mask = f['mask'][:]
        for i in range(10):
            plt.figure()
            plt.subplot(211)
            plt.imshow(label[i, :, :], 'gray')
            plt.subplot(212)
            plt.imshow(mask[i, :, :], 'gray')
            plt.show()
