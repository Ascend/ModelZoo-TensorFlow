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

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from data import multiinputs
from model import select_model, get_checkpoint
import os
import json

tf.app.flags.DEFINE_boolean('multitask', True,
	'Whether utilize multitask model')
tf.app.flags.DEFINE_string('model_type', 'LMTCNN-1-1',
	'choose model structure. LMTCNN-1-1 and mobilenet_multitask for multitask. inception, levi_hassner_bn and levi_hassner for singletask ')
tf.app.flags.DEFINE_string('class_type', '',
	'select which single task to train (Age or Gender), only be utilized when multitask=False and choose single task model_type')

tf.app.flags.DEFINE_string('eval_dir','./tfrecord1/train_val_test_per_fold_agegender/test_fold_is_0',
	'valiation and testing set directory')
tf.app.flags.DEFINE_string('eval_data','test',
	'data type validation or test')
tf.app.flags.DEFINE_string('model_dir','./models/train_val_test_per_fold_agegender/test_fold_is_0/LMTCNN-1-1-run-6016',
	'trained model directory')
tf.app.flags.DEFINE_string('result_dir','./results',
	'output files stored in results directory')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
	'what processing unit to execute inference on')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
	'number of preprocessing threads')
tf.app.flags.DEFINE_integer('num_examples', 10000,
	'number of examples to run')
tf.app.flags.DEFINE_integer('image_size', 227,
	'image size')
tf.app.flags.DEFINE_integer('batch_size', 128,
	'batch size')
tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
	'checkpoint basename')

FLAGS = tf.app.flags.FLAGS

def evaluate(saveresultdir):

	if FLAGS.multitask:

		with tf.Graph().as_default() as g:
			age_input_file = os.path.join(FLAGS.eval_dir, 'mdage.json')
			gender_input_file = os.path.join(FLAGS.eval_dir, 'mdgender.json')
			with open(age_input_file, 'r') as fage, open(gender_input_file, 'r') as fgender:
				mdage = json.load(fage)
				mdgender = json.load(fgender)

			if FLAGS.eval_data is 'validation':
				num_eval = mdage['valid_counts']
				tfpath = os.path.join(FLAGS.eval_dir, 'validation.tfrecord')
			else:
				num_eval = mdage['test_counts']
				tfpath = os.path.join(FLAGS.eval_dir, 'test.tfrecord')

			model_fn = select_model(FLAGS.model_type)
			dataset =  multiinputs(data_dir = tfpath, batch_size=FLAGS.batch_size,train=False,num_epochs= 1)
			with tf.device(FLAGS.device_id):
				print('Executing on %s' % FLAGS.device_id)
				images_holder = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,227,227,3])

				agelabels_holder = tf.placeholder(tf.int32,shape = [FLAGS.batch_size])
				genderlabels_holder = tf.placeholder(tf.int32,shape = [FLAGS.batch_size])
				agelogits, genderlogits = model_fn(mdage['nlabels'], images_holder,mdgender['nlabels'], images_holder, 1, False)
				summary_op = tf.summary.merge_all()
				summary_writer = tf.summary.FileWriter(saveresultdir,g)
				saver = tf.train.Saver()
				# run eval once.
				# saver: Saver
				# summary_writer: summary writer
				# top_k_op: top k op.
				# summary_op: summary op.
				agetop1 = tf.nn.in_top_k(agelogits, agelabels_holder, 1)
				agetop2 = tf.nn.in_top_k(agelogits, agelabels_holder, 2)

				gendertop1 = tf.nn.in_top_k(genderlogits, genderlabels_holder, 1)
				iterator = dataset.make_one_shot_iterator()
				images0, agelabels0, genderlabels0 = iterator.get_next()
				with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(allow_soft_placement = True))) as sess:
					checkpoint_path = FLAGS.model_dir
					model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)
					saver.restore(sess, model_checkpoint_path)

					# start the queue runners.
					coord = tf.train.Coordinator()
					try:
						threads = []
						for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
							threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

						num_steps = int(math.ceil(num_eval / FLAGS.batch_size))
						agetrue_count1 = agetrue_count2 = gendertrue_count1=0
						total_sample_count = num_steps * FLAGS.batch_size
						step = 0

						while step < num_steps and not coord.should_stop():

							images, agelabels_1, genderlabels_1 = sess.run([images0, agelabels0, genderlabels0])
							agelabels_1 = (np.reshape(agelabels_1, (FLAGS.batch_size))).astype(np.int32)
							genderlabels_1 = (np.reshape(genderlabels_1, (FLAGS.batch_size))).astype(np.int32)
							start_time = time.time()
							agev, agepredictions1, agepredictions2, genderv, genderpredictions1= sess.run([agelogits, agetop1, agetop2, genderlogits, gendertop1],
																								feed_dict = {images_holder:images,agelabels_holder: agelabels_1, genderlabels_holder:genderlabels_1})
							duration = time.time() - start_time
							sec_per_batch = float(duration)
							examples_per_sec = FLAGS.batch_size / sec_per_batch

							agetrue_count1 += np.sum(agepredictions1)
# 							print(agetrue_count1)
							agetrue_count2 += np.sum(agepredictions2)
                            
							gendertrue_count1 += np.sum(genderpredictions1)
							format_str = ('%s (%.1f examples/sec; %.3f sec/batch)')
							print(format_str % (datetime.now(), examples_per_sec, sec_per_batch))

							step += 1
                        
							if step == num_steps-1:
								agepredictions1 = agetrue_count1 / total_sample_count
								agepredictions2 = agetrue_count2 / total_sample_count
								genderpredictions1 = gendertrue_count1 / total_sample_count
								print('Age => %s: precision @ 1 = %.3f (%d/%d)' % (datetime.now(), agepredictions1, agetrue_count1, total_sample_count))
								print('Age => %s: precision @ 2 = %.3f (%d/%d)' % (datetime.now(), agepredictions2, agetrue_count2, total_sample_count))
								print('Gender => %s: precision @ 1 = %.3f (%d/%d)' % (datetime.now(), genderpredictions1, gendertrue_count1, total_sample_count))
								resulttxt = saveresultdir+os.sep+FLAGS.eval_data+'_result.txt'
								with open(resulttxt, 'w') as f:
									f.write('Age => %s: precision @ 1 = %.3f (%d/%d) \n' % (datetime.now(), agepredictions1, agetrue_count1, total_sample_count))
									f.write('Age => %s: precision @ 2 = %.3f (%d/%d) \n' % (datetime.now(), agepredictions2, agetrue_count2, total_sample_count))
									f.write('Gender => %s: precision @ 1 = %.3f (%d/%d) \n' % (datetime.now(), genderpredictions1, gendertrue_count1, total_sample_count))
									f.close()              

					except Exception as e:
						coord.request_stop(e)

					coord.request_stop()
					#coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):

	if not os.path.exists(FLAGS.result_dir):
		os.mkdir(FLAGS.result_dir)	
	folddirlist = FLAGS.model_dir.split(os.sep)
	saveresultdir = FLAGS.result_dir
	for i in range(len(folddirlist)):		
		saveresultdir = saveresultdir+os.sep+folddirlist[i]
		if not os.path.exists(saveresultdir):
			os.mkdir(saveresultdir)
	
	evaluate(saveresultdir)


if __name__ == '__main__':
	tf.app.run()
