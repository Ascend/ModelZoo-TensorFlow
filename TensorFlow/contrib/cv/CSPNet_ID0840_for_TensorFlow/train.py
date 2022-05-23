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

import os.path
import time
from models import cspdarknet53
import tensorflow as tf
from utils import load_cifar, random_batch, format_time
import numpy as np
import argparse

def train(
		epochs, 
		batch_size,
		learning_rate,
		momentum, 
		lmbda, 
		resume, 
		cifar_path,
		display_step, 
		val_epoch, 
		ckpt_path, 
		summary_path):

	Xtr, Ytr, Xte, Yte = load_cifar(cifar_path)
	Xtr, Ytr = random_batch(Xtr, Ytr, batch_size)
	Xte, Yte = random_batch(Xte, Yte, batch_size)
	ts_size = len(Ytr)
	num_batches = int(float(ts_size) / batch_size)
	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.float32, [None, 10])

	lr = tf.placeholder(tf.float32)

	pred = cspdarknet53.classifier(x)

	# cross-entropy and weight decay
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='cross-entropy'))
	
	with tf.name_scope('l2_loss'):
		l2_loss = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
		tf.summary.scalar('l2_loss', l2_loss)
	
	with tf.name_scope('loss'):
		loss = cross_entropy + l2_loss
		tf.summary.scalar('loss', loss)

	# accuracy
	with tf.name_scope('accuracy'):
		correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		tf.summary.scalar('accuracy', accuracy)
	
	global_step = tf.Variable(0, trainable=False)

	# momentum optimizer
	with tf.name_scope('optimizer'):
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(loss, global_step=global_step)

	# merge summaries to write them to file
	merged = tf.summary.merge_all()

	# checkpoint saver
	saver = tf.train.Saver()
	coord = tf.train.Coordinator()
	init = tf.global_variables_initializer()

	config = tf.ConfigProto()
	custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name =  "NpuOptimizer"
	custom_op.parameter_map["use_off_line"].b = True
	custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
	# config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

	with tf.Session(config=config) as sess:
		if resume:
			saver.restore(sess, ckpt_path)
		else:
			sess.run(init)
		train_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess.graph)

		start_time = time.time()
		num_steps = 0
		train_time = 0
		warnup_time = 0

		for e in range(0, epochs):
			if e==1:
				warnup_time = train_time
			for im, l in zip(Xtr, Ytr):
				iter_start_time = time.time()
				summary_str,_, step = sess.run([merged,optimizer, global_step], feed_dict={x: im, y: l, lr: learning_rate})
				iter_end_time = time.time()
				train_time += iter_end_time - iter_start_time

				train_writer.add_summary(summary_str, step)
				num_steps = step

				# display current training informations
				if step % display_step == 0:
					c, a = sess.run([loss, accuracy], feed_dict={x: im, y: l, lr: learning_rate})
					print ('Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}'.format(e, step, c, a))
					int_time = time.time()
					print ('Elapsed time: {}'.format(format_time(int_time - start_time)))

				# make test and evaluate validation accuracy
			if ((e+1) % val_epoch == 0) or (e+1==epochs):
					print('Epoch {}, validating ....'.format(e))
					v_a = []
					for i in range(len(Yte)):
						v_a.append(sess.run(accuracy, feed_dict={x: Xte[i], y: Yte[i], lr: learning_rate}))
					v_a = np.mean(v_a)
					# intermediate time
					int_time = time.time()
					print ('Elapsed time: {}'.format(format_time(int_time - start_time)))
					print ('Validation accuracy: {:.04f}'.format(v_a))

					save_path = saver.save(sess, os.path.join(ckpt_path, 'cspdarknet53.ckpt'))
					print('checkpoint saved in file: %s' % save_path)


		end_time = time.time()
		print ('Elapsed time: {}'.format(format_time(end_time - start_time)))
		StepTime = (train_time - warnup_time)/(num_steps*(epochs-1)/epochs)
		print('StepTime = {}'.format(StepTime))
		FPS = batch_size/ StepTime
		print('FPS = {}'.format(FPS))
		save_path = saver.save(sess, os.path.join(ckpt_path, 'cspdarknet53.ckpt'))
		print('checkpoint saved in file: %s' % save_path)

		coord.request_stop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--MOMENTUM', default=0.9, type=int)
	parser.add_argument('--LAMBDA', default=5e-04, type=float)
	parser.add_argument('--LEARNING_RATE', default=1e-03, type=float)
	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--SUMMARY', default='logs', type=str)
	parser.add_argument('--output_path', default='./checkpoint', type=str)
	parser.add_argument('--data_path', default='./dataset', type=str)
	parser.add_argument('--DISPLAY_STEP', default=1000, type=int)
	parser.add_argument('--VAL_EPOCH', default=5, type=int)
	parser.add_argument('--resume', default=False, type=bool)

	args = parser.parse_args()
	MOMENTUM = args.MOMENTUM
	LAMBDA = args.LAMBDA # for weight decay
	LEARNING_RATE = args.LEARNING_RATE
	EPOCHS = args.epochs
	BATCH_SIZE = args.batch_size
	CKPT_PATH = args.output_path
	if not os.path.exists(CKPT_PATH):
		os.makedirs(CKPT_PATH)
	SUMMARY = args.SUMMARY
	if not os.path.exists(SUMMARY):
		os.makedirs(SUMMARY)

	CIFAR_PATH = args.data_path
	DISPLAY_STEP = args.DISPLAY_STEP
	VAL_EPOCH = args.VAL_EPOCH
	resume = args.resume

	train(
		EPOCHS, 
		BATCH_SIZE, 
		LEARNING_RATE,
		MOMENTUM, 
		LAMBDA, 
		resume, 
		CIFAR_PATH,
		DISPLAY_STEP, 
		VAL_EPOCH, 
		CKPT_PATH, 
		SUMMARY)

