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
import os, argparse
import moxing as mox
import scipy.io
import math
import skimage.measure as measure

def psnr(target, ref, scale):
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]
    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)
def ssim(target, ref, scale):
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]
    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]
    _ssim = measure.compare_ssim(target_data, ref_data)
    return _ssim

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/vdsr-jcr/data/omtest/")#obs该目录下放原图（这里是3.mat）、放大2倍后的图（3_2.mat）、模型处理后的放大图（即对3_2.mat转bin文件用om模型处理得到的new0_output_0.txt）
parser.add_argument("--train_url", type=str, default="/vdsr-jcr/")
args = parser.parse_args()
data_dir = "/cache/dataset"
os.makedirs(data_dir)
mox.file.copy_parallel(args.data_url, data_dir)

a = []
b = []
mat_dict = scipy.io.loadmat(os.path.join(args.data_url, "3_2.mat"))#需要视输入调整
mat_dict1 = scipy.io.loadmat(os.path.join(args.data_url, "3.mat"))#需要视输入调整
input_img = mat_dict["img_2"]
img = mat_dict1["img_raw"]
predict = np.loadtxt(os.path.join(args.data_url, "new0_output_0.txt"))#需要视输入调整
a.append(input_img)
b.append(img)
a2 = a[0]
b2 = b[0]

#这三行中的276X276与3.mat形状对应，要随输入的不同而调整
input_y = np.resize(a2, (276, 276))
gt_y = np.resize(b2, (276, 276))
img_vdsr_y = np.resize(predict, (276, 276))

#这4行最后的参数（这里是2）要保持与放大倍数相同，即若是对原图放大三倍后的模糊图像进行处理则应改为3（注：放大倍数由文件名可知（下划线后），如3_2表示对3号原图放大两倍后的图像）
psnr_bicub = psnr(input_y, gt_y, 2)
psnr_vdsr = psnr(img_vdsr_y, gt_y, 2)
ssim_bicub = ssim(input_y, gt_y, 2)
ssim_vdsr = ssim(img_vdsr_y, gt_y, 2)

print ("PSNR: bicubic %f\tVDSR %f" % (psnr_bicub, psnr_vdsr))
print ("SSIM: bicubic %f\tVDSR %f" % (ssim_bicub, ssim_vdsr))

mox.file.copy_parallel(args.data_url, args.train_url)