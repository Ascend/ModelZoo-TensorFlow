#
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
#
from npu_bridge.npu_init import *
import os
import sys
from PIL import Image
from sets import Set
import numpy as np
import cv2
data_dir = sys.argv[1]
anno_dir = os.path.join(data_dir, 'Annotations/480p/')
save_dir = os.path.join(data_dir, 'Annotations/480p_split/')
save_dir_all = os.path.join(data_dir, 'Annotations/480p_all')
fds = os.listdir(anno_dir)
for fd in fds:
    im_list = os.listdir(os.path.join(anno_dir, fd))
    im_list = [item for item in im_list if item[-3:] == 'png']
    im = np.array(Image.open(os.path.join(anno_dir + fd, '00000.png')))
    cls_n = im.max()
    for item in im_list:
        im_path = os.path.join(anno_dir, fd, item)
        im = np.array(Image.open(im_path))
        all_dir = os.path.join(save_dir_all, fd)
        if not os.path.exists(all_dir):
            os.makedirs(all_dir)
        binary_map = (im > 0)
        mask_image = (binary_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(all_dir, item), mask_image)
        for i in range(1, cls_n+1):
            split_dir = os.path.join(save_dir, fd, str(i))
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            binary_map = im == i
            mask_image = (binary_map * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(split_dir, item), mask_image)

        


