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
# limitations under the License.import os
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
import math

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import sys

# target:目标图像  ref:参考图像
def PSNR(target, ref):
	if target.shape != ref.shape:
		raise ValueError('输入图像的大小应该一致！')

	diff = ref - target
	diff = diff.flatten('C')

	rmse = math.sqrt(np.mean(diff ** 2.))
	psnr = 20 * math.log10(np.max(target) / rmse)

	return psnr


def SSIM(target, ref, K1 = 0.01, K2 = 0.03, gaussian_kernel_sigma = 1.5, gaussian_kernel_width = 11, L = 255):
	# 高斯核，方差为1.5，滑窗为11*11
	gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
	for i in range(gaussian_kernel_width):
		for j in range(gaussian_kernel_width):
			gaussian_kernel[i, j] = (1 / (2 * math.pi * (gaussian_kernel_sigma ** 2))) * math.exp(
				-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

	target = np.array(target, dtype = np.float32)
	ref = np.array(ref, dtype = np.float32)
	if target.shape != ref.shape:
		raise ValueError('输入图像的大小应该一致！')

	target_window = convolve2d(target, np.rot90(gaussian_kernel, 2), mode = 'valid')
	ref_window = convolve2d(ref, np.rot90(gaussian_kernel, 2), mode = 'valid')

	mu1_sq = target_window * target_window
	mu2_sq = ref_window * ref_window
	mu1_mu2 = target_window * ref_window

	sigma1_sq = convolve2d(target * target, np.rot90(gaussian_kernel, 2), mode = 'valid') - mu1_sq
	sigma2_sq = convolve2d(ref * ref, np.rot90(gaussian_kernel, 2), mode = 'valid') - mu2_sq
	sigma12 = convolve2d(target * ref, np.rot90(gaussian_kernel, 2), mode = 'valid') - mu1_mu2

	C1 = (K1 * L) ** 2
	C2 = (K2 * L) ** 2
	ssim_array = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	ssim = np.mean(np.mean(ssim_array))

	return ssim
sum=0
ji=20*2.0
if len(sys.argv)!=2:
	exit(-1)
for i in range(20):
	target=np.load(sys.argv[1]+'_'+str(i)+'_2_0.npy')
	ref=np.load(sys.argv[1]+'_'+str(i)+'_1_0.npy')
	sum=sum+PSNR(target,ref)
	target=np.load(sys.argv[1]+'_'+str(i)+'_2_1.npy')
	ref=np.load(sys.argv[1]+'_'+str(i)+'_1_1.npy')
	sum=sum+PSNR(target,ref)	
print(sys.argv[1],end='')
print(sum/ji)
#import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
#from PIL import Image
#import os
	## 获取指定路径下所有指定后缀的文件
## dir 指定路径
## ext 指定后缀，链表&不需要带点 或者不指定。例子：['xml', 'java']

#def list_allfile(path,all_files=[],all_py_files=[]):    
	#if os.path.exists(path):
		#files=os.listdir(path)
	#else:
		#print('this path not exist')
	#for file in files:
		#if os.path.isdir(os.path.join(path,file)):
			#list_allfile(os.path.join(path,file),all_files)
		#else:
			#all_files.append(os.path.join(path,file))
	#for file in all_files:
		#if file.endswith('.npy'):
			#all_py_files.append(file)
	#return all_py_files
##print(files)
#files=list_allfile('.')
#for i in range(len(files)):

	#print(files[i])
	#if files[i].split('.')[2]=='py':
		#continue
	#data=np.load(files[i])
	##plt.axis('off')   
	##plt.imshow(data,cmap='Greys_r')
	##plt.savefig(str(i)+'.png',bbox_inches='tight')
	#data=((data*2-1)+1)*127.5
	##im=Image.fromarray(np.uint8(data))
	##print(im.size)
	##im.save(str(i)+'_.png')


	

	#psnr = PSNR(target, ref)
	#print('PSNR为:{}'.format(psnr))

	##ssim = SSIM(target, ref)
	##print('SSIM为:{}'.format(ssim))
