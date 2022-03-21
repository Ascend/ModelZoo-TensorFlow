#
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
#
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import math

class Linear(object):
    def __init__(self, h_in, out_len, trainable=True):
        in_len = np.prod(h_in.get_shape().as_list()[1:])
        initial = tf.truncated_normal([in_len, out_len], stddev = 0.1)
        self.W = tf.Variable(initial, trainable=trainable)
        self.b = tf.Variable(tf.constant(0.1, shape=[out_len]), trainable=trainable)
        self.h_out = tf.matmul(tf.reshape(h_in, [-1,in_len]), self.W) + self.b


class RegLinear(Linear):
    def __init__(self, Linear, FMo=None, trainable=False):
        shape = Linear.W.get_shape().as_list()
        self.W = tf.Variable(tf.constant(0.0, shape=shape), trainable=trainable)
        self.b = tf.Variable(tf.constant(0.0, shape=[shape[-1]]), trainable=trainable)

        if FMo == None:
            self.reg_obj = tf.reduce_sum(tf.square(Linear.W - self.W))
            self.reg_obj += tf.reduce_sum(tf.square(Linear.b - self.b)) 
        else:
            self.reg_obj = tf.reduce_sum(tf.mul(FMo.W,tf.square(Linear.W - self.W)))
            self.reg_obj += tf.reduce_sum(tf.mul(FMo.b,tf.square(Linear.b - self.b)))


class DropLinear(Linear):
    def __init__(self, h_in, out_len, keep_prob, trainable=True):
        in_len = np.prod(h_in.get_shape().as_list()[1:])
        initial = tf.truncated_normal([in_len, out_len], stddev = 0.1)
        self.W = tf.Variable(initial,trainable=trainable)
        self.b = tf.Variable(tf.constant(0.1,shape=[out_len]),trainable=trainable)
        
        self.dropbase = RegLinear(self)

        self.h_out = npu_ops.dropout(tf.matmul(h_in, self.W) + self.b, keep_prob)\
            + tf.matmul(h_in, self.dropbase.W) + self.dropbase.b

