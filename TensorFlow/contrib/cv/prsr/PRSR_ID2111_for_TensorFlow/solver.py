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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
from ops import *
from data import *
from net import *
from utils import *
import os
import time

# #------------------11.4 Dump数据采集------------------------
# # 引用precision_tool/tf_config.py
# import argparse
# import precision_tool.tf_config as npu_tf_config
# import moxing as mox
# import precision_tool.config as CONFIG
# #------------------11.4 Dump数据采集------------------------


flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
	def __init__(self):
		self.device_id = conf.device_id
		self.train_dir = conf.train_dir
		self.samples_dir = conf.samples_dir
		if not os.path.exists(self.train_dir):
			os.makedirs(self.train_dir)
		if not os.path.exists(self.samples_dir):
			os.makedirs(self.samples_dir)
			# datasets params
		self.num_epoch = conf.num_epoch
		self.batch_size = conf.batch_size
		# optimizer parameter
		self.learning_rate = conf.learning_rate
		if conf.use_gpu:
			device_str = '/gpu:' + str(self.device_id)
		else:
			device_str = '/cpu:0'
		# with tf.device(device_str):
		# dataset
		self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size)
		self.net = Net(self.dataset.hr_images, self.dataset.lr_images, 'prsr')
		# optimizer
		self.global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0),
		                                   trainable = False)
		learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
		                                           500000, 0.5, staircase = True)
		optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.95, momentum = 0.9, epsilon = 1e-8)
		self.train_op = optimizer.minimize(self.net.loss, global_step = self.global_step)

	def train(self):
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		summary_op = tf.summary.merge_all()
		saver = tf.train.Saver()
		# Create a session for running operations in the Graph.
		# config = tf.ConfigProto(allow_soft_placement=True)
		# config.gpu_options.allow_growth = True

		# ----------------------------2021.9.11 NPU-------------------------------------
		config = tf.ConfigProto()

		# # ------------------11.4 Dump数据采集------------------------
		# config = npu_tf_config.session_dump_config(config, action = 'dump')
		# # ------------------11.4 Dump数据采集------------------------

		custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
		custom_op.name = "NpuOptimizer"

		# ----------------------2021.9.16 NPU-------------
		custom_op.parameter_map["mix_compile_mode"].b = True
		# ----------------------2021.9.16 NPU-------------

		# # ----------------------2021.10.31 NPU-------------
		# custom_op.parameter_map["use_off_line"].b = True
		# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
		# # ----------------------2021.10.31 NPU-------------

		config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
		config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
		# ----------------------------2021.9.11 NPU-------------------------------------

		sess = tf.Session(config = config)

		# Initialize the variables (like the epoch counter).
		sess.run(init_op)
		# saver.restore(sess, './models/model.ckpt-30000')
		summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		iters = 0
		try:
			while not (coord.should_stop() | iters == 380):
				# Run training steps or whatever
				t1 = time.time()
				_, loss = sess.run([self.train_op, self.net.loss], feed_dict = {self.net.train: True})
				t2 = time.time()
				print('step %d, loss = %.2f  %.1f examples/sec  %.3f sec/batch' % (
				(iters, loss, self.batch_size / (t2 - t1), (t2 - t1))))
				iters += 1
				if iters % 10 == 0:
					summary_str = sess.run(summary_op, feed_dict = {self.net.train: True})
					summary_writer.add_summary(summary_str, iters)
				if iters % 1000 == 0:
					# self.sample(sess, mu=1.0, step=iters)
					self.sample(sess, mu = 1.1, step = iters)
					# self.sample(sess, mu=100, step=iters)
				if iters % 10000 == 0:
					checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step = iters)
				# if iters % 370 == 0:
				# 	parser = argparse.ArgumentParser()
				# 	parser.add_argument("--train_url", type = str, default = "./output")
				# 	config2, unparsed = parser.parse_known_args()
					#mox.file.copy_parallel(CONFIG.ROOT_DIR, config2.train_url)
					#mox.file.copy_parallel('/home/ma-user/modelarts/user-job-dir/code', config2.train_url)
		except tf.errors.OutOfRangeError:
			checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path)
			print('Done training -- epoch limit reached')
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()

		# Wait for threads to finish.
		coord.join(threads)
		sess.close()

	def sample(self, sess, mu = 1.1, step = None):
		c_logits = self.net.conditioning_logits
		p_logits = self.net.prior_logits
		lr_imgs = self.dataset.lr_images
		hr_imgs = self.dataset.hr_images
		np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
		gen_hr_imgs = np.zeros((self.batch_size, 32, 32, 3), dtype = np.float32)
		# gen_hr_imgs = np_hr_imgs
		# gen_hr_imgs[:,16:,16:,:] = 0.0
		np_c_logits = sess.run(c_logits, feed_dict = {lr_imgs: np_lr_imgs, self.net.train: False})
		print('iters %d: ' % step)

		for i in range(32):
			for j in range(32):
				for c in range(3):
					np_p_logits = sess.run(p_logits, feed_dict = {hr_imgs: gen_hr_imgs})
					new_pixel = logits_2_pixel_value(
						np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256],
						mu = mu)
					gen_hr_imgs[:, i, j, c] = new_pixel
		#
		save_samples(np_lr_imgs, self.samples_dir + '/lr_' + str(mu * 10) + '_' + str(step) + '.jpg')
		save_samples(np_hr_imgs, self.samples_dir + '/hr_' + str(mu * 10) + '_' + str(step) + '.jpg')
		save_samples(gen_hr_imgs, self.samples_dir + '/generate_' + str(mu * 10) + '_' + str(step) + '.jpg')