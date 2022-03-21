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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops 
import os
import moxing as mox

best_auc = 0.0

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--EMBEDDING_DIM', type=int, default=18) 
    parser.add_argument('--HIDDEN_SIZE', type=int, default=18*2) 
    parser.add_argument('--ATTENTION_SIZE', type=int, default=18*2) 
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--test_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--model_type', type=str, default='DIN') 
    parser.add_argument('--model_path', type=str, default='s3://din/din-output/save/')
    parser.add_argument('--best_model_path', type=str, default='/test-0115/din/dnn_best_model/ckpt_noshuff') 
    parser.add_argument('--l2_reg', type=float, default=0.0) 
    parser.add_argument('--random_seed', type=int, default=2018) 
    parser.add_argument('--data_url', type=str, default='/test-0115/din/data')
    parser.add_argument('--train_url', type=str, default='../models/')
 
    return parser.parse_args()


def prepare_data(input, target, maxlen = None, return_neg = False):
	# x: a list of sentences

	lengths_x = [len(s[4]) for s in input]
	seqs_mid = [inp[3] for inp in input]
	seqs_cat = [inp[4] for inp in input]
	noclk_seqs_mid = [inp[5] for inp in input]
	noclk_seqs_cat = [inp[6] for inp in input]

	if maxlen is not None:
		new_seqs_mid = []
		new_seqs_cat = []
		new_noclk_seqs_mid = []
		new_noclk_seqs_cat = []
		new_lengths_x = []
		for l_x, inp in zip(lengths_x, input):
			if l_x > maxlen:
				new_seqs_mid.append(inp[3][l_x - maxlen:])
				new_seqs_cat.append(inp[4][l_x - maxlen:])
				new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
				new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
				new_lengths_x.append(maxlen)
			else:
				new_seqs_mid.append(inp[3])
				new_seqs_cat.append(inp[4])
				new_noclk_seqs_mid.append(inp[5])
				new_noclk_seqs_cat.append(inp[6])
				new_lengths_x.append(l_x)
		lengths_x = new_lengths_x
		seqs_mid = new_seqs_mid
		seqs_cat = new_seqs_cat
		noclk_seqs_mid = new_noclk_seqs_mid
		noclk_seqs_cat = new_noclk_seqs_cat

		if len(lengths_x) < 1:
			return None, None, None, None

	n_samples = len(seqs_mid)
	#maxlen_x = numpy.max(lengths_x)
	maxlen_x = 100 #max_length
	neg_samples = len(noclk_seqs_mid[0][0])

	mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
	cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
	noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
	noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
	mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
	for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
		mid_mask[idx, :lengths_x[idx]] = 1.
		mid_his[idx, :lengths_x[idx]] = s_x[:maxlen_x]
		cat_his[idx, :lengths_x[idx]] = s_y[:maxlen_x]
		noclk_mid_his[idx, :lengths_x[idx], :] = no_sx[:maxlen_x]
		noclk_cat_his[idx, :lengths_x[idx], :] = no_sy[:maxlen_x]

	uids = numpy.array([inp[0] for inp in input])
	mids = numpy.array([inp[1] for inp in input])
	cats = numpy.array([inp[2] for inp in input])

	if return_neg:
		return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x), noclk_mid_his, noclk_cat_his

	else:
		return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)

def eval(sess, test_data, model, model_path):

	loss_sum = 0.
	accuracy_sum = 0.
	aux_loss_sum = 0.
	nums = 0
	maxlen = args.maxlen
	stored_arr = []
	for src, tgt in test_data:
		nums += 1
		uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
		if uids.shape[0] != args.batch_size:
			continue
		prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats])
		loss_sum += loss
		aux_loss_sum = aux_loss
		accuracy_sum += acc
		prob_1 = prob[:, 0].tolist()
		target_1 = target[:, 0].tolist()
		for p ,t in zip(prob_1, target_1):
			stored_arr.append([p, t])
	test_auc = calc_auc(stored_arr)
	accuracy_sum = accuracy_sum / nums
	loss_sum = loss_sum / nums
	aux_loss_sum / nums
	global best_auc
	if best_auc < test_auc:
		best_auc = test_auc
		model.save(sess, model_path)
	return test_auc, loss_sum, accuracy_sum, aux_loss_sum

def train(args, seed = 2):

	train_file = args.data_url + "/local_train_splitByUser"
	test_file = args.data_url + "/local_test_splitByUser"
	uid_voc = args.data_url + "/uid_voc.pkl"
	mid_voc = args.data_url + "/mid_voc.pkl"
	cat_voc = args.data_url + "/cat_voc.pkl"
	item_info = args.data_url + '/item-info'
	reviews_info = args.data_url + '/reviews-info'
	batch_size = args.batch_size
	maxlen = args.maxlen
	test_iter = args.test_iter
	save_iter = args.save_iter
	model_type = args.model_type
	model_path = args.model_path #args.data_url + model_type + str(seed)
	best_model_path = args.data_url + model_type + str(seed)
	config = tf.ConfigProto()
	custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = "NpuOptimizer"
	custom_op.parameter_map["use_off_line"].b = True
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
	# gpu_options = tf.GPUOptions(allow_growth=True)
	with tf.Session(config=config) as sess:
		train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen, shuffle_each_epoch=False)
		test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen)
		n_uid, n_mid, n_cat = train_data.get_n()
		if model_type == 'DNN':
			model = Model_DNN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'PNN':
			model = Model_PNN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.args.ATTENTION_SIZE)
		elif model_type == 'Wide':
				model = Model_WideDeep(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN':
			model = Model_DIN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-att-gru':
			model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-gru-att':
			model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-qa-attGru':
			model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-vec-attGru':
			model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIEN':
				model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		else:
			print ("Invalid model_type : %s", model_type)
			return

		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		start_time = time.time()
		iter = 0
		lr = 0.001
		for itr in range(args.epoch):
			loss_sum = 0.0
			accuracy_sum = 0.
			aux_loss_sum = 0.
			for src, tgt in train_data:
				uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
				if uids.shape[0] != args.batch_size:
					continue
				loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats])
				loss_sum += loss
				accuracy_sum += acc
				aux_loss_sum += aux_loss
				iter += 1
				sys.stdout.flush()
				if (iter % test_iter) == 0:
					print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- tran_aux_loss: %.4f' % \
										  (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
					#print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, best_model_path))
					loss_sum = 0.0
					accuracy_sum = 0.0
					aux_loss_sum = 0.0
				if (iter % save_iter) == 0:
					print('save model iter: %d' %(iter))
					model.save(sess, model_path+'model.ckpt')
			lr *= 0.5

def test(args, seed = 2):

	train_file = args.data_url + "/local_train_splitByUser"
	test_file = args.data_url + "/local_test_splitByUser"
	uid_voc = args.data_url + "/uid_voc.pkl"
	mid_voc = args.data_url + "/mid_voc.pkl"
	cat_voc = args.data_url + "/cat_voc.pkl"
	item_info = args.data_url + '/item-info'
	reviews_info = args.data_url + '/reviews-info'
	batch_size = args.batch_size
	maxlen = args.maxlen
	model_type = args.model_type

	model_path = args.model_path #args.data_url + model_type + str(seed)
	config = tf.ConfigProto()
	custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = "NpuOptimizer"
	custom_op.parameter_map["use_off_line"].b = True
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

	# gpu_options = tf.GPUOptions(allow_growth=True)
	with tf.Session(config=config) as sess:
		train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen)
		test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen)
		n_uid, n_mid, n_cat = train_data.get_n()
		if model_type == 'DNN':
			model = Model_DNN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'PNN':
			model = Model_PNN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'Wide':
			model = Model_WideDeep(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN':
			model = Model_DIN(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-att-gru':
			model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-gru-att':
			model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-qa-attGru':
			model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIN-V2-gru-vec-attGru':
			model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		elif model_type == 'DIEN':
			model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, args.EMBEDDING_DIM, args.HIDDEN_SIZE, args.ATTENTION_SIZE)
		else:
			print ("Invalid model_type : %s", model_type)
			return

		model.restore(sess, model_path+'model.ckpt')
		print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, model_path))

if __name__ == '__main__':
	args = parse_args()

	#if len(sys.argv) == 4:
	#	SEED = int(sys.argv[3])
	#else:
	#	SEED = 3
	#tf.set_random_seed(SEED)
	#numpy.random.seed(SEED)
	#random.seed(SEED)

	# if sys.argv[1] == 'train':
	train(args)
	# elif sys.argv[1] == 'test':
	test(args)
	# else:
	# 	print('do nothing...')


