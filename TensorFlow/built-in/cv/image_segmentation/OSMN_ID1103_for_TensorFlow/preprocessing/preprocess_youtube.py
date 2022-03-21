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
import numpy as np
import sys
from PIL import Image
import shutil
base_dir = sys.argv[1]
im_dir = os.path.join(base_dir, 'Images')
label_dir = os.path.join(base_dir, 'Labels')
list_file = os.path.join(base_dir, 'all.txt')
fds = [fd for fd in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, fd))]
list_objs=[]
for fd in fds:
    prin(fd)
    sub_fds = os.listdir(os.path.join(base_dir, fd, 'data'))
    sub_fds = sorted(sub_fds)
    for sub_fd in sub_fds:
        list_objs.append(fd+'_'+sub_fd)
        im_list = os.listdir(os.path.join(base_dir, fd,'data',sub_fd,'shots','001','images'))
        new_im_fd = os.path.join(im_dir, fd+'_'+sub_fd)
        new_label_fd = os.path.join(label_dir, fd+'_'+sub_fd)
        if not os.path.exists(new_im_fd):
            os.makedirs(new_im_fd)
            os.makedirs(new_label_fd)
        for im_name in im_list:
            shutil.copy(os.path.join(base_dir, fd,'data',sub_fd,'shots','001','images',im_name), 
                    os.path.join(new_im_fd, im_name))
            label = Image.open(os.path.join(base_dir, fd,'data',sub_fd,'shots','001','labels',im_name[:-4]+'.jpg'))
            label = Image.fromarray(((np.array(label)>127) * 255).astype(np.uint8))
            label.save(os.path.join(new_label_fd, im_name))
with open(list_file,'w') as f:
    f.write('\n'.join(list_objs))

