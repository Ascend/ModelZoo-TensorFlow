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
import os
import tensorflow as tf 
import numpy as np
import pickle

def weight(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.01)
	w = tf.Variable(initial, name=name)
	tf.add_to_collection('weights', w)
	return w

def bias(value, shape, name):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
	return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
	return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)

def lrn(x, depth_radius, bias, alpha, beta):
	return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def relu(x):
	return tf.nn.relu(x)

def batch_norm(x):
	epsilon = 1e-3
	batch_mean, batch_var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)

def onehot(index):
	""" It creates a one-hot vector with a 1.0 in
		position represented by index 
	"""
	onehot = np.zeros(10)
	onehot[index] = 1.0
	return onehot

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		X = dict[b'data']
		Y = dict[b'labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
		Y = np.array(Y)
		return X,Y

def load_cifar(ROOT):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT,'data_batch_%d' % b)
		X ,Y = unpickle(f)
		xs.append(X)
		ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	Xte,Yte = unpickle(os.path.join(ROOT,'test_batch'))
	Yte = [onehot(x) for x in Yte]
	Ytr = [onehot(x) for x in Ytr]
	return Xtr,Ytr,Xte,Yte


def random_batch(x, y, batch_size):
	sz = len(x)
	idx = np.arange(0, sz)
	np.random.shuffle(idx)
	idx = idx[:sz-sz % batch_size]
	x = np.split(x[idx], sz // batch_size)
	y = np.split(np.array(y)[idx], sz // batch_size)
	return x, y

def format_time(time):
	""" It formats a datetime to print it

		Args:
			time: datetime

		Returns:
			a formatted string representing time
	"""
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))



