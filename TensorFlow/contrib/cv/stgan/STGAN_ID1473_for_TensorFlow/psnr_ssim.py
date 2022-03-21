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
import os
import os.path as path
from PIL import Image
import numpy as np
import math
# from skimage.measure import compare_psnr as psnr
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', help='experiment_name')
args_ = parser.parse_args()
experiment_name = args_.experiment_name

root = './output/%s/sample_testing' % experiment_name

sum_psnr = []
sum_ssim = []
cnt = 0
for im in os.listdir(root):
    if cnt % 1000 == 0:
        print(cnt)
    img = Image.open(path.join(root, im))
    src = np.array(img.crop((0, 0, 128, 128)))
    dst = np.array(img.crop((140, 0, 268, 128)))
    sum_psnr.append(psnr(src, dst))
    sum_ssim.append(ssim(src, dst, multichannel=True))
    cnt += 1
print('--psnr: {} --'.format(sum(sum_psnr) / float(cnt)))
print('--ssim: {} --'.format(sum(sum_ssim) / float(cnt)))
with open('psnr.txt', 'w') as f:
    f.write('%f\n%f\n' % (sum(sum_psnr) / float(cnt), sum(sum_ssim) / float(cnt)))
    f.write('\n'.join(['%f %f' % (sum_psnr[i], sum_ssim[i]) for i in range(cnt)]))
    f.close()
