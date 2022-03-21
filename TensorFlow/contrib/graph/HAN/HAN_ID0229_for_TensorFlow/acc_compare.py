# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import struct
import os
import numpy as np
import tensorflow as tf

import scipy.io as sio
def load_data_dblp(path='/home/wei/program/tmp/HAN/data/ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels = data['label']
    return truelabels


def ReadFile():
    filepath_1='om/output/20210603_131312/HAN_output_0.bin' # om生成的bin
    filepath_2='predication_bin/ckpt_prediction.bin' # ckpt生成的bin
    # filepath_3='pb_prediction1.bin'

    test_1 = np.fromfile(filepath_1, dtype=np.float32)
    test_2 = np.fromfile(filepath_2, dtype=np.float32)
    # test_3 = np.fromfile(filepath_3, dtype=np.float32)

    a_1 = np.reshape(test_1, (3025, 3))
    a_2 = np.reshape(test_2, (3025, 3))
    print(a_1)
    print(a_2)

    true_label = load_data_dblp()  # 真实label

    # correct_prediction = np.equal(np.argmax(a_1, 1), np.argmax(true_label, 1))
    # print(correct_prediction)
    #
    # n = 0
    # n_all = 0
    # n_test = 0
    # for i in correct_prediction:
    #     n_all = n_all + 1
    #     # test
    #     if n_all > 300 and n_all < 1060:
    #         n_test = n_test + 1
    #         if i == True:
    #             n = n + 1
    #     if n_all > 1361 and n_all < 2025:
    #         n_test = n_test + 1
    #         if i == True:
    #             n = n + 1
    #     if n_all > 2326 and n_all < 3024:
    #         n_test = n_test + 1
    #         if i == True:
    #             n = n + 1
    # print("ckpt acc :", n/n_test)

    correct_prediction = np.equal(np.argmax(a_2, 1), np.argmax(true_label, 1))
    n = 0
    n_all = 0
    n_test = 0
    for i in correct_prediction:
        n_all = n_all + 1
        # test
        if n_all > 300 and n_all < 1060:
            n_test = n_test + 1
            if i == True:
                n = n + 1
        if n_all > 1361 and n_all < 2025:
            n_test = n_test + 1
            if i == True:
                n = n + 1
        if n_all > 2326 and n_all < 3024:
            n_test = n_test + 1
            if i == True:
                n = n + 1
    print("om acc :", n/n_test)

if __name__ == '__main__':
	ReadFile()
