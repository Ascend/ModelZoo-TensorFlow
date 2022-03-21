"""LICENSE"""
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
from __future__ import print_function
from npu_bridge.npu_init import *

import datetime
import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import sys
import scipy.ndimage
import argparse

BATCH_SIZE = 24
EPOCHES = 1
LOGGING = 40
#LOGGING = 1
#MODEL_SAVE_PATH = './model/'
IS_TRAINING = True
#MODEL_SAVE_PATH = './model/'


#f = h5py.File('Training_Dataset.h5', 'r')
#sources = f['data'][:]
#sources = np.transpose(sources, (0, 3, 2, 1))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, default='', help='data_path')
	parser.add_argument('--output_path', type=str, default='', help='output_path')
	parser.add_argument('--n_batches', type=int, default=954, help='n_batches')
	return parser.parse_args()

"""main"""
def main():
	args = parse_args()
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		with open(args.output_path + 'logs.txt', 'a') as logs:  # 设置文件对象
			logs.write(('\n=============Begin to train the network===========TrainingTime: %s\n' % (datetime.datetime.now())))  # 将字符串

		f = h5py.File(args.data_path + '/Training_Dataset.h5', 'r')
		# # for key in f.keys():
		# #   print(f[key].name)
		sources = f['data'][:]
		sources = np.transpose(sources, (0, 3, 2, 1))

		train(sources, args.output_path, EPOCHES, BATCH_SIZE, logging_period=LOGGING, n_batches=args.n_batches)
		#train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		#path = './test_imgs/'
		#savepath = './results/'
		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)

		Time=[]

		for dirpath, dirnames, filenames in os.walk(args.data_path + "model20211111/"):
		#for dirpath, dirnames, filenames in os.walk(MODEL_SAVE_PATH):
			for dir in dirnames:
				dirPathFull = os.path.join(dirpath, dir)
				model_path = dirPathFull + '/' + dir + '.ckpt'

				#savepath = './results/'+dir+'/'
				savepath = args.output_path + '/results/' + dir + '/'
				if not os.path.exists(savepath):
					os.makedirs(savepath)
				for i in range(20):
					index = i + 1
					#ir_path = path + 'IR' + str(index) + '.bmp'
					#vis_path = path + 'VIS' + str(index) + '.bmp'
					ir_path = args.data_path + '/test_imgs/IR/IR' + str(index) + '.bmp'
					vis_path = args.data_path + '/test_imgs/VIS/VIS' + str(index) + '.bmp'
					begin = time.time()
					#model_path = MODEL_SAVE_PATH + 'model.ckpt'
					generate(ir_path, vis_path, model_path, index, args.output_path)
					end = time.time()
					Time.append(end - begin)
					print("pic_num:%s" % index)
				print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


if __name__ == '__main__':
	#数据集路径
	#data_path = sys.argv[1]
	# 模型路径
	#模型路径--图片路径
	#output_path = sys.argv[2]

	main()

