"""
ckpt_to_pb
"""
# coding=utf-8
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow  as tf
from tensorflow_core.python.tools import freeze_graph
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from stnet import spatial_transformer_network as transformer
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import meta_graph
import numpy as np
import argparse
# %% Load ckpt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='the training data')
parser.add_argument('--output_path', type=str, default='./output/',
                    help='the path model saved')
args, unkown = parser.parse_known_args()
def ckpt2pb():
    """
    ckpt2pb
    Returns:

    """
    tf.reset_default_graph()
    input_checkpoint = os.path.join(args.ckpt_path + 'model.ckpt')
    # ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    # input_checkpoint = ckpt.model_checkpoint_path #得ckpt文件路径

    # saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=True)
    # graph = tf.get_default_graph()  # 获得默认的图
    # input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        
        x = tf.placeholder(tf.float32, [None, 1600])
        x_tensor = tf.reshape(x, [-1, 40, 40, 1])
        W_fc_loc1 = weight_variable([1600, 20])
        b_fc_loc1 = bias_variable([20])
        W_fc_loc2 = weight_variable([20, 6])
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
        h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
        h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)
        out_size = (40, 40)
        h_trans = transformer(x_tensor, h_fc_loc2, out_size)
        filter_size = 3
        n_filters_1 = 16
        W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
        b_conv1 = bias_variable([n_filters_1])
        h_conv1 = tf.nn.relu(
            tf.nn.conv2d(input=h_trans,
                            filter=W_conv1,
                            strides=[1, 2, 2, 1],
                            padding='SAME') +
            b_conv1)
        n_filters_2 = 16
        W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
        b_conv2 = bias_variable([n_filters_2])
        h_conv2 = tf.nn.relu(
            tf.nn.conv2d(input=h_conv1,
                            filter=W_conv2,
                            strides=[1, 2, 2, 1],
                            padding='SAME') +
            b_conv2)
        h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])
        n_fc = 1024
        W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
        b_fc1 = bias_variable([n_fc])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
        W_fc2 = weight_variable([n_fc, 10])
        b_fc2 = bias_variable([10])
        y_logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        y_logits = tf.identity(y_logits, 'output_logit')
        output_node_names="output_logit"
        tf.io.write_graph(sess.graph_def, args.output_path, 'temp.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
		        input_graph=os.path.join(args.output_path, 'temp.pb'),   # 传入write_graph生成的模型文件
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=input_checkpoint,  # 传入训练生成的checkpoint文件
		        output_node_names='output_logit',  # 与重新定义的推理网络输出节点保持一致
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph=os.path.join(args.output_path, 'stnet.pb'),   # 改为需要生成的推理网络的名称
		        clear_devices=False,
		        initializer_nodes='')
        #print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    ckpt2pb()
