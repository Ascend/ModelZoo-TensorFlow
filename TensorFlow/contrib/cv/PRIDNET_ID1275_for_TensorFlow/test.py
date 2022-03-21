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
from scipy.io import loadmat
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import importlib
import matplotlib.pyplot as plt
import tensorflow  as tf
flags=tf.flags
Flags=flags.FLAGS
flags.DEFINE_string('val_dir1',"result",'train model result')
flags.DEFINE_string('val_dir2','dataset','train model dataset url')

val_dir1=Flags.val_dir1
val_dir2=Flags.val_dir2
#val_dir1='D:/Noisy raw-RGB data/ValidationGtBlocksRaw.mat'
#val_dir2='D:/4000/ValidationCleanBlocksRaw.mat'
mat1 = loadmat(val_dir1)
mat2 = loadmat(val_dir2)

B1=mat1['ValidationGtBlocksRaw']
B2=mat2['results']
a,_=compare_ssim(B1,B2,full=True)     #SSIM
print('SSIM: ',a)
#print(a)
print('-----------------------------------')

psnr =compare_psnr(B1, B2)               #PSNR
print('PSNR: ',psnr)
