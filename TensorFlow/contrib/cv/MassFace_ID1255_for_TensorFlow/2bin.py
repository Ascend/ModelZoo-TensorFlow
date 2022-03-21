
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
from PIL import Image as im
import imageio
from lib import lfw
import os
#import moxing as mx
from lib import utils

#lfw_dir = r"C:\Users\94582\Desktop\MassFac\dataset\lfw-112x112"
#lfw_pairs = r"C:\Users\94582\Desktop\MassFac\dataset/pairs.txt"
#result_dir = r"C:\Users\94582\Desktop\MassFac\dataset\lfw-bin"
lfw_dir = "/cache/user-job-dir/MassFac/dataset/lfw-112x112"
lfw_pairs = "/cache/user-job-dir/MassFac/dataset/pairs.txt"
result_dir = "/cache/user-job-dir/MassFac/dataset/lfw-bin"

if(os.path.isdir(result_dir) == False):
    os.mkdir(result_dir)


alltot = 0
tot = 0
size = 1

out_put = np.zeros([size,112,112,3],dtype=np.float32)
pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs)
nrof_samples = len(paths)
def get_image(path):
    clear_image = im.open(path)
    clear_image_arr = (np.array(clear_image).astype('float32') - 127.5) / 128.0
    clear_image.close()
    return clear_image_arr
for i in range(nrof_samples):
    clear_image_arr = get_image(paths[i])
    out_put[0] = clear_image_arr
    out_put.tofile(result_dir + '/{}.bin'.format(str(i)))
with open(os.path.join(result_dir ,'actual_issame.txt'),'w') as file:
    for issame in actual_issame:
        file.write( '1 ' if issame else '0 ')

mx.file.copy_parallel('/cache/user-job-dir/MassFac/dataset/lfw-bin/', 'obs://qyy-unet/lfw-bin/')

print('ok')