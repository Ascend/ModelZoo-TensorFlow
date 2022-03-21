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
from scipy import misc
from PIL import Image
import tensorflow as tf
import glob, os, re
from PSNR import psnr
from PSNR import ssim
import scipy.io
import pickle
from MODEL import model
# from MODEL_FACTORIZED import model_factorized
import time
#import moxing as mox
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--model_path")
parser.add_argument("--data_url", type=str, default="./data/test/")
#parser.add_argument("--train_url", type=str, default="/vdsr-jcr/")
# args = parser.parse_args()
args = parser.parse_args()
#model_path = args.model_path
# data_dir = "/cache/dataset"
# os.makedirs(data_dir)
#mox.file.copy_parallel(args.data_url, data_dir)

#DATA_PATH = os.path.join(args.data_url, "Set5")
#DATA_PATH = os.path.join(args.data_url, "Set14")
#DATA_PATH = os.path.join(args.data_url, "B100")
DATA_PATH = os.path.join(args.data_url, "Urban100")
# 选择复现第99个模型
model_ckpt = os.path.join("./checkpointC", "VDSR_adam_epoch_099.ckpt-96000")   #add

def get_img_list(data_path):
	l = glob.glob(os.path.join(data_path, "*"))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat", 2])
			if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat", 3])
			if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat", 4])
	return train_list
def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset + batch_size]
	input_list = []
	gt_list = []
	scale_list = []
	for pair in target_list:
		print (pair[1])
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		#if mat_dict.has_key("img_2"):
		if "img_2" in mat_dict:
			input_img = mat_dict["img_2"]
		elif "img_3" in mat_dict:
			input_img = mat_dict["img_3"]
		elif "img_4" in mat_dict:
			input_img = mat_dict["img_4"]
		else:
			continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img)
		gt_list.append(gt_img)
		scale_list.append(pair[2])
	return input_list, gt_list, scale_list
def test_VDSR_with_sess(epoch, ckpt_path, data_path, sess):
	# 选择SET5和SET14数据还是B100、urban100数据
	folder_list = glob.glob(data_path)
	print ('folder_list', folder_list)
	saver.restore(sess, ckpt_path)
	psnr_dict = {}
	ssim_dict = {}
	for folder_path in folder_list:
		psnr_list = []
		ssim_list = []
		img_list = get_img_list(folder_path)
		for i in range(len(img_list)):
			input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
			input_y = input_list[0]
			gt_y = gt_list[0]
			start_t = time.time()
			img_vdsr_y = sess.run([output_tensor], feed_dict={
				input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
			img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
			end_t = time.time()
			print ("end_t", end_t, "start_t", start_t)
			print ("time consumption", end_t - start_t)
			print ("image_size", input_y.shape)

			# 增加代码
			psnr_bicub = psnr(input_y, gt_y, scale_list[0])
			psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
			ssim_bicub = ssim(input_y, gt_y, scale_list[0])
			ssim_vdsr = ssim(img_vdsr_y, gt_y, scale_list[0])

			print ("PSNR: bicubic %f\tVDSR %f" % (psnr_bicub, psnr_vdsr))
			psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])
			print ("SSIM: bicubic %f\tVDSR %f" % (ssim_bicub, ssim_vdsr))
			ssim_list.append([ssim_bicub, ssim_vdsr, scale_list[0]])

		psnr_dict[os.path.basename(folder_path)] = psnr_list
		ssim_dict[os.path.basename(folder_path)] = ssim_list

		# 额外增加的代码
		# 输出PSNR
		bicubic_psnr_2 = [x[0] for x in psnr_list[::3]]
		bicubic_psnr_3 = [x[0] for x in psnr_list[1::3]]
		bicubic_psnr_4 = [x[0] for x in psnr_list[2::3]]
		bicubic_mean_2 = np.mean(bicubic_psnr_2)
		bicubic_mean_3 = np.mean(bicubic_psnr_3)
		bicubic_mean_4 = np.mean(bicubic_psnr_4)

		# 		print "bicubic_LIST_2:",bicubic_list_2
		# 		print "bicubic_LIST_3:",bicubic_list_3
		# 		print "bicubic_LIST_4:",bicubic_list_4
		print ("bicubic_mean_psnr_2:", bicubic_mean_2)
		print ("bicubic_mean_psnr_3:", bicubic_mean_3)
		print ("bicubic_mean_psnr_4:", bicubic_mean_4)

		VDSR_psnr_2 = [y[1] for y in psnr_list[::3]]
		VDSR_psnr_3 = [y[1] for y in psnr_list[1::3]]
		VDSR_psnr_4 = [y[1] for y in psnr_list[2::3]]
		VDSR_mean_2 = np.mean(VDSR_psnr_2)
		VDSR_mean_3 = np.mean(VDSR_psnr_3)
		VDSR_mean_4 = np.mean(VDSR_psnr_4)

		# 		print "VDSR_list_2:",VDSR_list_2
		# 		print "VDSR_list_3:",VDSR_list_3
		# 		print "VDSR_list_4:",VDSR_list_4
		print ("VDSR_mean_psnr_2:", VDSR_mean_2)
		print ("VDSR_mean_psnr_3:", VDSR_mean_3)
		print ("VDSR_mean_psnr_4:", VDSR_mean_4)

		# 输出SSIM
		bicubic_ssim_2 = [x[0] for x in ssim_list[::3]]
		bicubic_ssim_3 = [x[0] for x in ssim_list[1::3]]
		bicubic_ssim_4 = [x[0] for x in ssim_list[2::3]]
		bicubic_mean_ssim_2 = np.mean(bicubic_ssim_2)
		bicubic_mean_ssim_3 = np.mean(bicubic_ssim_3)
		bicubic_mean_ssim_4 = np.mean(bicubic_ssim_4)

		# 		print "bicubic_LIST_2:",bicubic_list_2
		# 		print "bicubic_LIST_3:",bicubic_list_3
		# 		print "bicubic_LIST_4:",bicubic_list_4
		print ("bicubic_ssim_2:", bicubic_mean_ssim_2)
		print ("bicubic_ssim_3:", bicubic_mean_ssim_3)
		print ("bicubic_ssim_4:", bicubic_mean_ssim_4)

		VDSR_ssim_2 = [y[1] for y in ssim_list[::3]]
		VDSR_ssim_3 = [y[1] for y in ssim_list[1::3]]
		VDSR_ssim_4 = [y[1] for y in ssim_list[2::3]]
		VDSR_mean_ssim_2 = np.mean(VDSR_ssim_2)
		VDSR_mean_ssim_3 = np.mean(VDSR_ssim_3)
		VDSR_mean_ssim_4 = np.mean(VDSR_ssim_4)

		# 		print "VDSR_list_2:",VDSR_list_2
		# 		print "VDSR_list_3:",VDSR_list_3
		# 		print "VDSR_list_4:",VDSR_list_4
		print ("VDSR_ssim_2:", VDSR_mean_ssim_2)
		print ("VDSR_ssim_3:", VDSR_mean_ssim_3)
		print ("VDSR_ssim_4:", VDSR_mean_ssim_4)
def test_VDSR(epoch, ckpt_path, data_path):
	with tf.Session() as sess:
		test_VDSR_with_sess(epoch, ckpt_path, data_path, sess)

if __name__ == '__main__':
	model_list = sorted(glob.glob(model_ckpt))
	print (model_list)
	model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
	print (model_list)
	tf.reset_default_graph()
	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
		shared_model = tf.make_template('shared_model', model)
		output_tensor, weights = shared_model(input_tensor) #jcr即为离线推理产生的bin文件
		# 		output_tensor, weights 	= model(input_tensor)
		saver = tf.train.Saver(weights)
		# 		saver = tf.train.import_meta_graph("/home/ma-user/work/checkpointA/VDSR_const_clip_0.01_epoch_000.ckpt-3364.meta")
		tf.initialize_all_variables().run()

		print ("Testing model", model_ckpt)
		test_VDSR_with_sess(80, model_ckpt, DATA_PATH, sess)