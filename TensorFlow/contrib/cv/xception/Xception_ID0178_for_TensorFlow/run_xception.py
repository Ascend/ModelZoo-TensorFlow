# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
import os
import numpy as np
import sys
import time
#import moxing as mox
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

from xception_model import XceptionModel
from  data_utils import get_train_data,get_test_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path", None,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_string(
    "output_path", None,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_bool(
	"do_train", False, 
	"Whether to run training.")

flags.DEFINE_integer(
    "image_num", 50000,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_integer(
    "class_num", 1000,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_integer(
	"batch_size", 64,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_integer(
    "epoch", 10,
    "The config json file corresponding to the pre-trained ALBERT model. ")

flags.DEFINE_float(
	"learning_rate", 0.001, 
	"The initial learning rate for GradientDescent.")

flags.DEFINE_integer(
	"save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")






def training_op( log,label):
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=label, name='entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    # accuracy
    correct = tf.nn.in_top_k(log, label, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op = optimizer.minimize(loss)
        return op,loss,accuracy


def evaluation_op(log, label):
 
    # return???accuracy, loss The average accuracy and loss of the current step              

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=label, name='entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(log, label,1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return  loss, accuracy


def tf_data_list(tf_data_path):
	filepath = tf_data_path
	tf_data_list = []
	file_list = os.listdir(filepath)
	if len(file_list)==0:
		raise Exception("?????????tfrecord????????????????????????")
	for i in file_list:
		tf_data_list.append(os.path.join(filepath,i))
	return tf_data_list

def  validate_flags_or_throw():
	
	if os.path.exists(FLAGS.data_path)==False: #
		raise Exception(FLAGS.data_path,"????????????????????????????????????????????????tfrecord????????????????????????")
	
	if FLAGS.do_train:
		if os.path.exists(FLAGS.output_path)==False: #??????????????????????????????????????????
			os.mkdir(FLAGS.output_path)
			print(FLAGS.output_path,"??????????????????")  		
	else:
		if FLAGS.model_path == "None":
			raise Exception("eval??????????????????????????????????????????")

def main(_):
	validate_flags_or_throw()
	inputx = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 299, 299, 3], name="inputx")
	inputy = tf.placeholder(tf.int64, name="inputy")

	if FLAGS.do_train==True:
		out = XceptionModel(inputx, FLAGS.class_num,True, data_format='channels_last')
		train_op, train_loss, train_val = training_op(out, inputy)
		images_batch ,labels_batch =  get_train_data(tf_data_list(FLAGS.data_path), FLAGS.batch_size, FLAGS.epoch)

	if FLAGS.do_train==False:
		out = XceptionModel(inputx, FLAGS.class_num,False, data_format='channels_last')
		test_loss, test_acc = evaluation_op(out, inputy)
		images_batch ,labels_batch =  get_test_data(tf_data_list(FLAGS.data_path), FLAGS.batch_size)

	config = tf.ConfigProto(allow_soft_placement=True)
	custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = "NpuOptimizer"
	custom_op.parameter_map["use_off_line"].b = True  # ?????????AI?????????????????????
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # ??????remap??????
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	total_time = 0

	if FLAGS.model_path != "None":
		model = FLAGS.model_path
		saver.restore(sess, model)


	try:
		#???????????? train
		if FLAGS.do_train==True:	
			for epoch in range(FLAGS.epoch):
				for step in range(int(FLAGS.image_num / FLAGS.batch_size)):
					begin_time = time.time()
					x_in, y_in = sess.run([images_batch, labels_batch])
					y_in = np.squeeze(y_in, 1)
					_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_val],
					                            feed_dict={inputx: x_in, inputy: y_in})
					end_time = time.time() - begin_time
					total_time = total_time + end_time
					if (step + 1) % 10 == 0:
						dt = total_time / 10
						img_per_sec = FLAGS.batch_size / dt
						total_time = 0
						print('Epoch %d step %d train loss = %.4f FPS = %7.1f train accuracy = %.2f%%' % (
						epoch + 1, step + 1, tra_loss, img_per_sec, tra_acc * 100.0))	
					if (step+1)%FLAGS.save_checkpoints_steps==0:
						checkpoint_path = os.path.join(FLAGS.output_path, "xception_model.ckpt")
						saver.save(sess, checkpoint_path)
		#???????????? eval
		if FLAGS.do_train==False:
			acc = []
			loss = []
			for step in range(int(FLAGS.image_num / FLAGS.batch_size)):
				x_in, y_in = sess.run([images_batch, labels_batch])
				y_in = np.squeeze(y_in, 1)
				test_losss, test_accs = sess.run([test_loss, test_acc], feed_dict={inputx: x_in, inputy: y_in})
				acc.append(test_accs)
				loss.append(test_losss)
			print("xception eval  loss=%.5f   acc=%.4f "%(np.mean(loss),np.mean(acc)))            
	except tf.errors.OutOfRangeError:
		print('epoch limit reached')
	finally:
		print("Done")
		sess.close()

if __name__ == '__main__':
	tf.app.run()