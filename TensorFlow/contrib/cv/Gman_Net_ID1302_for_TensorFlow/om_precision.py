"""
om precision
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

pred_path = '/home/ma-user/modelarts/inputs/data_url_0/2021129_18_17_24_494196/'
clear_path = '/home/ma-user/modelarts/inputs/data_url_0/CleanImages/TestImages/'


def cal_psnr(im1, im2):
    """

    Args:
        im1:
        im2:

    Returns:

    """
    mse = ((im1.astype(np.float32) - im2.astype(np.float32)) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


psnr_list = []
for i in range(14000):
    pred = np.fromfile(pred_path + '{}_output_0.bin'.format(i), dtype='float32').reshape([224, 224, 3])
    clear = np.fromfile(clear_path + '{}.bin'.format(i), dtype='float32').reshape([224, 224, 3])
    tmp_psnr = cal_psnr(pred, clear)
    psnr_list.append(tmp_psnr)
    print('The PSNR of {}-th image is: {} dB'.format(i, tmp_psnr))
print('*' * 20)
print('Average PSNR: {}'.format(np.mean(psnr_list)))
