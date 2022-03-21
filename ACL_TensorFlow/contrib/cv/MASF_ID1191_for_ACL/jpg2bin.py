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
# ===========================
#   Author      : ChenZhou
#   Time        : 2021/11
#   Language    : Python
# ===========================

import cv2
import os
dst_path = 'E:/PACS/binfile/giraffe'
x = 0
for root, dirs, files in os.walk('E:/PACS/kfold/cartoon/giraffe'):
    for d in dirs:
        print(d)  # 打印子资料夹的个数
    for file in files:
        print(file)
        # 讀入圖像
        img_path = root + '/' + file
        img = cv2.imread(img_path, 1)
        img = img.astype('float32')
        print(img_path)
        img.tofile(dst_path + "/" + file + ".bin")





