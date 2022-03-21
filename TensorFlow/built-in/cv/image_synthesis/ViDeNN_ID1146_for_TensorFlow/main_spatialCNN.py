# -*- coding: utf-8 -*-
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
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""
from npu_bridge.npu_init import *

import argparse
from glob import glob
import sys
import tensorflow as tf
import os
from model_spatialCNN import denoiser
from utilis import *
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./ckpt', help='checkpoints are saved here')
parser.add_argument('--save_dir', dest='save_dir', default='./data/denoised', help='denoised sample are saved here')
args = parser.parse_args()

def sortKeyFunc(s):
	 return int(os.path.basename(s)[:-4])
	 
def denoiser_train(denoiser):
	with load_data('./data/train/img_clean_pats.npy') as data_:
		data = data_
	with load_data('./data/train/img_noisy_pats.npy') as data_noisy_:
		data_noisy = data_noisy_

	noisy_eval_files = glob('./data/test/noisy/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_data_noisy = load_images(noisy_eval_files)
	eval_files = glob('./data/test/original/*.png')
	eval_files = sorted(eval_files)

	eval_data = load_images(eval_files)
	#npu modify begin
	#denoiser.train(data, eval_data[0:20], eval_data_noisy[0:20], hooks=npu_hooks_append(hooks_list=data_noisy), batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)
	denoiser.train(data, data_noisy, eval_data, eval_data_noisy, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch)
	#npu modify end


def denoiser_test(denoiser):
	noisy_eval_files = glob('./data/test/noisy/*.jpg')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('./data/test/original/*.jpg')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir=args.save_dir)

def denoiser_for_temp3_training(denoiser):
	noisy_eval_files = glob('../Temp3-CNN/data/train/noisy/*/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('../Temp3-CNN/data/train/original/*/*.png')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir='../Temp3-CNN/data/train/denoised/')


def main(_):
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

#	lr = args.lr * np.ones([args.epoch])
#	lr[3:] = lr[0] / 10.0
	#npu modify begin 
	global_config_dynamic = tf.ConfigProto()
	custom_op = global_config_dynamic.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = 'NpuOptimizer'
	custom_op.parameter_map["use_off_line"].b = True
	custom_op.parameter_map["dynamic_input"].b = True
	custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
	custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("data:[1,20~800,20~800,3],[],[1,20~800,20~800,3]")
	global_config_dynamic.graph_options.rewrite_options.remapping = RewriterConfig.OFF

	global_config_static = tf.ConfigProto()
	custom_op_ = global_config_static.graph_options.rewrite_options.custom_optimizers.add()
	custom_op_.name = 'NpuOptimizer'
	custom_op_.parameter_map["use_off_line"].b = True
	global_config_static.graph_options.rewrite_options.remapping = RewriterConfig.OFF

	#npu modify end
	if args.use_gpu:
		# Control the gpu memory setting per_process_gpu_memory_fraction
		print("GPU\n")
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
		#npu modify begin
		#with tf.Session(config =npu_config_proto(config_proto= tf.ConfigProto(gpu_options=gpu_options))) as sess:
		sess_static = tf.Session(config=global_config_static)
		sess_dynamic = tf.Session(config=global_config_dynamic)
		#npu modify end
		model = denoiser(sess_static, sess_dynamic, args.lr)
		if args.phase == 'train':
			denoiser_train(model)
		elif args.phase == 'test':
			denoiser_test(model)
		elif args.phase == 'test_temp':
			denoiser_for_temp3_training(model)
		else:
			print('[!] Unknown phase')
			exit(0)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			#npu modify begin
			#with tf.Session(config=npu_config_proto()) as sess:
			with tf.Session(config=global_config) as sess:
			#npu modify end
				model = denoiser(sess)
				if args.phase == 'train':
					denoiser_train(model, lr=lr)
				elif args.phase == 'test':
					denoiser_test(model)
				elif args.phase == 'test_temp':
					denoiser_for_temp3_training(model)
				else:
					print('[!] Unknown phase')
					exit(0)


if __name__ == '__main__':
	tf.app.run()

