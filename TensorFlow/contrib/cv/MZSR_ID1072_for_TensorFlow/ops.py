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

def conv2d(x, kernel, bias, strides=1, scope=None, activation=None):
    with tf.variable_scope(scope):
        out = tf.nn.conv2d(x,kernel,[1,strides,strides,1],padding='SAME', name='conv2d')
        out = tf.nn.bias_add(out,bias, name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)

def dense(x, weights, bias, scope=None, activation=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out=tf.matmul(x, weights,name='dense')
        out=tf.nn.bias_add(out,bias,name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)