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
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as npimport

import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.1


""" Discriminator1 """ 
class Discriminator1(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.weight_vars.append(self._create_variables(1, 16, 3, scope = 'conv1'))
			self.weight_vars.append(self._create_variables(16, 32, 3, scope = 'conv2'))
			self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv3'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)
    
	""" discrim """ 
	def discrim(self, img, reuse):
		conv_num = len(self.weight_vars)
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		for i in range(conv_num):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
			else:
				out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = True,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
		out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				out = tf.layers.dense(out, 1, activation = tf.nn.tanh, use_bias = True, trainable = True,
				                      reuse = reuse)
		out = out / 2 + 0.5
		return out


""" conv2d_1 """ 
def conv2d_1(x, kernel, bias, strides, use_relu=True, use_BN=True, Scope=None, Reuse=None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out


""" Discriminator2 """ 
class Discriminator2(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.weight_vars.append(self._create_variables(1, 16, 3, scope = 'conv1'))
			self.weight_vars.append(self._create_variables(16, 32, 3, scope = 'conv2'))
			self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv3'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	""" discrim """ 
	def discrim(self, img, reuse):
		conv_num = len(self.weight_vars)
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		for i in range(conv_num):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
			else:
				out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = True,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
		out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				out = tf.layers.dense(out, 1, activation = tf.nn.tanh, use_bias = True, trainable = True,
				                      reuse = reuse)
		out = out / 2 + 0.5
		return out


""" conv2d_2 """ 
def conv2d_2(x, kernel, bias, strides, use_relu=True, use_BN=True, Scope=None, Reuse=None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out

