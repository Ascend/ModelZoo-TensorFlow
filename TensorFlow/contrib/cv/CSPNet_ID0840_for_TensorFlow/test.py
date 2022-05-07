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
# Copyright 2022 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *
import os.path
import tensorflow as tf
from utils import load_cifar, random_batch
from models import cspdarknet53
import argparse


def test(
		batch_size,
		cifar_path,
		ckpt_path):

	Xtr, Ytr, Xte, Yte = load_cifar(cifar_path)
	Xte, Yte = random_batch(Xte, Yte, batch_size)
	
	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.float32, [None, 10])

	pred = cspdarknet53.classifier(x)

	# accuracy
	top1_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	top1_accuracy = tf.reduce_mean(tf.cast(top1_correct, tf.float32))

	saver = tf.train.Saver()

	config = tf.ConfigProto()
	custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name =  "NpuOptimizer"
	custom_op.parameter_map["use_off_line"].b = True
	custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

	with tf.Session(config=config) as sess:

		saver.restore(sess, os.path.join(ckpt_path))
		total_top1_accuracy = 0.

		for i in range(len(Yte)):
			top1_a = sess.run([top1_accuracy], feed_dict={x: Xte[i], y: Yte[i]})
			total_top1_accuracy += top1_a[0]
		print ('---- Final accuracy ----')
		print ('Top-1: {:.4f}'.format(total_top1_accuracy / len(Yte)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--CKPT_PATH', default='./checkpoint/cspdarknet53.ckpt', type=str)
	parser.add_argument('--CIFAR_PATH', default='./dataset', type=str)
	parser.add_argument('--DISPLAY_STEP', default=10, type=int)
	args = parser.parse_args()

	BATCH_SIZE = 64
	DISPLAY_STEP = args.DISPLAY_STEP
	CIFAR_PATH = args.CIFAR_PATH
	CKPT_PATH = args.CKPT_PATH

	test(
		BATCH_SIZE,
		CIFAR_PATH,
		CKPT_PATH)



