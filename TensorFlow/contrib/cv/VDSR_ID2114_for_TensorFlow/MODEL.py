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


import tensorflow as tf
import numpy as np

def model(input_tensor):

		weights = []
		tensor = None

		#conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
		conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

		for i in range(18):
			#conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
			conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
			conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
		
		#conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
		conv_w = tf.get_variable("conv_20_w", [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
		conv_b = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
		weights.append(conv_w)
		weights.append(conv_b)
		tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

		tensor = tf.add(tensor, input_tensor)
		return tensor, weights
