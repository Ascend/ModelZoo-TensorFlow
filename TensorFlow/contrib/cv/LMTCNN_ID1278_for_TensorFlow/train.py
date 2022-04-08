""" Rude Carnie: Age and Gender Deep Learning with Tensorflow found at
https://github.com/dpressel/rude-carnie
"""
# ==============================================================================
# MIT License
#
# Modifications copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from six.moves import xrange
from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
from data import multiinputs
from model import select_model
import json
import re
import sys
code_dir = os.path.dirname(__file__)
work_path = os.path.join(code_dir ,'../')
sys.path.append(work_path)
#import moxing as mox
from pdb import set_trace as bp

LAMBDA = 0.01
MOM = 0.9

tf.app.flags.DEFINE_boolean('multitask', True, 'Whether utilize multitask model')
tf.app.flags.DEFINE_string('model_type', 'LMTCNN-1-1','choose model structure. LMTCNN and mobilenet_multitask for multitask. inception, levi_hassner_bn and levi_hassner for singletask ')
tf.app.flags.DEFINE_string('class_type', '','select which single task to train (Age or Gender), only be utilized when multitask=False and choose single task model_type')

tf.app.flags.DEFINE_string('pre_checkpoint_path', '','if specified, restore this pretrained model before beginning any training.')
tf.app.flags.DEFINE_string('data_dir','./tfrecord/train_val_test_per_fold_agegender/test_fold_is_0','training age and gender directory.')
tf.app.flags.DEFINE_string('model_dir','./models','store models before training')
tf.app.flags.DEFINE_boolean('log_device_placement', False,'Whether to log device placement.')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'Number of preprocessing threads')
tf.app.flags.DEFINE_string('optim', 'Momentum','Optimizer')
tf.app.flags.DEFINE_integer('image_size', 227, 'Image size')
tf.app.flags.DEFINE_float('eta', 0.01,'Learning rate')
tf.app.flags.DEFINE_float('pdrop', 0.,'Dropout probability')
tf.app.flags.DEFINE_integer('max_steps', 50000,'Number of iterations')
tf.app.flags.DEFINE_integer('steps_per_decay', 10000,'Number of steps before learning rate decay')
tf.app.flags.DEFINE_float('eta_decay_rate', 0.1, 'learning rate decay')
tf.app.flags.DEFINE_integer('epochs', -1,'Number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 32,'Batch size')
tf.app.flags.DEFINE_string('checkpoint', 'checkpoint','Checkpoint name')
#tf.app.flags.DEFINE_string('data_url', './dataset', "data_root")
tf.app.flags.DEFINE_string('train_url', './log', "output_root")
# inception_v3.ckpt
tf.app.flags.DEFINE_string('pre_model','False', 'checkpoint file')

FLAGS = tf.app.flags.FLAGS

def exponential_staircase_decay(at_step=10000, decay_rate=0.1):

	print('decay [%f] every [%d] steps' % (decay_rate, at_step))

	def _decay(lr, global_step):
		return tf.train.exponential_decay(lr, global_step, at_step, decay_rate, staircase=True)

	return _decay

def optimizer(optim, eta, loss_fn, at_step, decay_rate):

	global_step = tf.Variable(0, trainable=False)
	optz = optim
	if optim == 'Adadelta':
		optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
		lr_decay_fn = None
	elif optim == 'Momentum':
		optz = lambda lr: tf.train.MomentumOptimizer(lr, MOM)
		lr_decay_fn = exponential_staircase_decay(at_step, decay_rate)

	return tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=4., learning_rate_decay_fn=lr_decay_fn)

def loss(logits, labels):

	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits, labels=labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	losses = tf.get_collection('losses')
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
	tf.summary.scalar('tl (raw)', total_loss)
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	loss_averages_op = loss_averages.apply(losses + [total_loss])
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))
	with tf.control_dependencies([loss_averages_op]):
		total_loss = tf.identity(total_loss)

	return total_loss

def multiloss(agelogits, agelabels, genderlogits, genderlabels):

	agelabels = tf.cast(agelabels, tf.int32)
	genderlabels = tf.cast(genderlabels, tf.int32)

	age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=agelogits, labels=agelabels, name='cross_entropy_per_example_age')
	gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=genderlogits, labels=genderlabels, name='cross_entropy_per_example_gender')

	age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy, name='cross_entropy_age')
	gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy, name='cross_entropy_gender')

	tf.add_to_collection('agelosses', age_cross_entropy_mean)
	tf.add_to_collection('genderlosses', gender_cross_entropy_mean)

	agelosses = tf.get_collection('agelosses')
	genderlosses = tf.get_collection('genderlosses')
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	totallosses = age_cross_entropy_mean+gender_cross_entropy_mean+LAMBDA*sum(regularization_losses)
	tf.summary.scalar('tl total (raw)', totallosses)

	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	loss_averages_op = loss_averages.apply(agelosses+genderlosses+[totallosses])

	for l in agelosses+genderlosses+[totallosses]:
		tf.summary.scalar(l.op.name + '(raw)', l)
		tf.summary.scalar(l.op.name , loss_averages.average(l))
	with tf.control_dependencies([loss_averages_op]):
		totallosses=tf.identity(totallosses)

	return agelosses, genderlosses, totallosses

def main(argv=None):

	if not os.path.exists(FLAGS.model_dir):
		os.mkdir(FLAGS.model_dir)
	folddirlist = FLAGS.data_dir.split(os.sep)
	subdir = FLAGS.model_dir+os.sep+folddirlist[-2]
	if not os.path.exists(subdir):
		os.mkdir(subdir)
	savemodeldir = subdir+os.sep+folddirlist[-1]
	if not os.path.exists(savemodeldir):
		os.mkdir(savemodeldir)

	if FLAGS.multitask:

		with tf.Graph().as_default():
			model_fn = select_model(FLAGS.model_type)
			# Open the metadata file and figure out nlabels, and size of epoch
			input_file_age = os.path.join(FLAGS.data_dir, 'mdage.json')
			input_file_gender = os.path.join(FLAGS.data_dir, 'mdgender.json')
			with open(input_file_age,'r') as fage:
				mdage = json.load(fage)
			with open(input_file_gender,'r') as fgender:
				mdgender = json.load(fgender)
			with tf.device('/cpu:0'):
				images_holder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 227, 227, 3])
				# agelabels_holder = tf.placeholder(tf.int64,shape = [FLAGS.batch_size],name = 'agelabels_holder')
				# genderlabels_holder = tf.placeholder(tf.int64,shape = [FLAGS.batch_size],name = 'genderlabels_holder')

				agelabels_holder = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
				genderlabels_holder = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

			agelogits, genderlogits = model_fn(mdage['nlabels'], images_holder, mdgender['nlabels'], images_holder,1 - FLAGS.pdrop, True)
			agelosses, genderlosses, totallosses = multiloss(agelogits, agelabels_holder, genderlogits,genderlabels_holder)
			agegendertrain_op = optimizer(FLAGS.optim, FLAGS.eta, totallosses, FLAGS.steps_per_decay,FLAGS.eta_decay_rate)

			saver = tf.train.Saver(tf.global_variables())
			summary_op = tf.summary.merge_all()

			sess = tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(allow_soft_placement = True, log_device_placement=FLAGS.log_device_placement)))
		

			tf.global_variables_initializer().run(session=sess)
			
			# fine-tune dp_multitask and mobilenet_multitask
			if FLAGS.pre_checkpoint_path:
				print('Trying to restore checkpoint from %s ' % FLAGS.pre_checkpoint_path)

				if FLAGS.model_type is 'LMTCNN':
					all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="multitaskdpcnn")
				elif FLAGS.model_type is 'mobilenet_multitask':
					all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="MobileNetmultitask")

				age_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="ageoutput")
				gender_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="genderoutput")
				all_variables.extend(age_variables)
				all_variables.extend(gender_variables)
				restorer = tf.train.Saver(all_variables)
				restorer.restore(sess, FLAGS.pre_checkpoint_path)

				print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.pre_checkpoint_path))

			#run_dir = '%s/%s-run-%d' %(savemodeldir, FLAGS.model_type, os.getpid())
			run_dir = '%s/%s-run' %(savemodeldir, FLAGS.model_type)
			checkpoint_path = '%s/%s' % (run_dir, FLAGS.checkpoint)
			if tf.gfile.Exists(run_dir) is False:
				print('Creating %s' % run_dir)
				tf.gfile.MakeDirs(run_dir)

			tf.train.write_graph(sess.graph_def, run_dir, 'agegendermodel.pb', as_text=True)
			tf.train.start_queue_runners(sess=sess)
			summary_writer = tf.summary.FileWriter(run_dir, sess.graph)

			steps_per_train_epoch = int(mdage['train_counts'] / FLAGS.batch_size)
			num_steps = FLAGS.max_steps if FLAGS.epochs < 1 else FLAGS.epochs * steps_per_train_epoch
			print('Requested number of steps [%d]' % num_steps)

			dataset = multiinputs(data_dir=os.path.join(FLAGS.data_dir, 'train.tfrecord'), batch_size=FLAGS.batch_size,train=True, num_epochs=FLAGS.epochs)
			iterator = dataset.make_one_shot_iterator()
			images0, agelabels0, genderlabels0 = iterator.get_next()
			for step in range(num_steps):
				#start_time = time.time()

				images, agelabels_1, genderlabels_1 = sess.run([images0, agelabels0, genderlabels0])
				# images1 = (np.reshape(images,(FLAGS.batch_size,227,227,3))).astype(np.float32)
				agelabels_1 = (np.reshape(agelabels_1, (FLAGS.batch_size))).astype(np.int32)
				genderlabels_1 = (np.reshape(genderlabels_1, (FLAGS.batch_size))).astype(np.int32)
                start_time = time.time()
				_,totallossvalue, agelossvalue, genderlossvalue = sess.run([ agegendertrain_op, totallosses, agelosses, genderlosses],
																		   feed_dict = {images_holder:images,agelabels_holder: agelabels_1, genderlabels_holder:genderlabels_1})
				duration = time.time() - start_time

				assert not np.isnan(agelossvalue), 'Model diverged with ageloss = NaN'
				assert not np.isnan(genderlossvalue), 'Model diverged with genderloss = NaN'
				assert not np.isnan(totallossvalue), 'Model diverged with totallossvalue= NaN'

				if step % 10 == 0:
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)

					format_str = ('%s: step %d , ageloss= %.3f , genderloss= %.3f , totalloss= %.3f (%.1f examples/sec ; %.3f ' 'sec/batch)')
					print(format_str % (datetime.now(), step, agelossvalue[0], genderlossvalue[0], totallossvalue, examples_per_sec, sec_per_batch))

				# loss evaluated every 100 steps
				if step % 100 == 0:
					summary_str = sess.run(summary_op,feed_dict = {images_holder:images,agelabels_holder: agelabels_1, genderlabels_holder:genderlabels_1})
					summary_writer.add_summary(summary_str, step)

				if step % 1000 == 0 or (step+1) == num_steps:
					saver.save(sess, checkpoint_path, global_step=step)
			#mox.file.copy_parallel(FLAGS.model_dir, FLAGS.train_url)


if __name__ == '__main__':
	os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = "0"
	
	try:
		tf.app.run()
	finally:
		print('''download log''')
