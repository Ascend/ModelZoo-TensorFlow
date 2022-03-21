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

import cv2
import numpy as np
import os


# compare accuracy with bin
def NME(S_bbox, infer, label):
    if label.shape[1] == 2:
        error = np.mean(np.sqrt((label[:, 0] - infer[:, 0]) ** 2 + (label[:, 1] - infer[:, 1]) ** 2))
    elif label.shape[1] == 3:
        error = np.mean(np.sqrt(
            (label[:, 0] - infer[:, 0]) ** 2 + (label[:, 1] - infer[:, 1]) ** 2 + (label[:, 2] - infer[:, 2]) ** 2))
    nme = error / S_bbox
    return nme*1000


uv_kpt_ind = np.loadtxt('Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
msameoutdir = 'output_bin/AFLW2000-3D'
binfiles = os.listdir(msameoutdir)
labeldir = 'Dataset/TestData/AFLW2000-3D'
output_num = 0
total = 0
for file in binfiles:
    if file.endswith(".bin"):
        output_num += 1
        tmp = np.fromfile(msameoutdir + '/' + file, dtype='float32')
        tmp = np.reshape(tmp, [1, 256, 256, 3])
        tmp = np.squeeze(tmp)
        tmp = tmp * 256 * 1.1
        inf = tmp[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
        pic_name = str(file.split(".jpg")[0]) + ".jpg"
        label = np.loadtxt(labeldir + '/' + pic_name[0: -17] + '.txt')
        nme_error = NME(256*256, inf, label)
        total += nme_error
        print("nme_error: %f%% \t picture_name: %s" % (nme_error, pic_name[0: -17]))

mean_nme_error = total / output_num
print("Total pic num: %d, mean_nme_error: %f%%" % (output_num, mean_nme_error))
