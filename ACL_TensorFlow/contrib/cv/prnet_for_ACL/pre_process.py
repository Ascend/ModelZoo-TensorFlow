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


# jpg convert to bin
src_path = 'Dataset/TestData/AFLW2000-3D'
dst_path = 'Dataset/TestData_bin/AFLW2000-3D'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
files = os.listdir(src_path)
for file in files:
    if file.endswith('.jpg'):
        src = src_path + "/" + file
        print("start to process %s" % src)
        img_org = cv2.imread(src)  # 读入数据
        img_org = img_org / 255.
        img_org = img_org.astype(np.float32)
        res = img_org[np.newaxis, :, :, :]  # 对原始图片进行需要的预处理
        print(res.dtype)
        res.tofile(dst_path + "/" + file[0: -4] + ".bin")  # 处理后的图片保存为bin文件

