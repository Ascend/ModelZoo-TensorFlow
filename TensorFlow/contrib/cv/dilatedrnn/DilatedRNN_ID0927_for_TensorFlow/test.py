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

# Testing Process

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from dilated_rnn.classification_models import drnn_classification

from npu_bridge.npu_init import *
from logger import my_logger

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='LSTM')
opt = parser.parse_args()

# model config
cell_type = opt.type
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 9
dilations = [1,2,4,8,16,32,64,128,256]
assert(len(hidden_structs) == len(dilations))

data_dir = "./dataset"
n_steps = 28*28
input_dims = 1
n_classes = 10 
test_results = []

# permutation seed 
seed = 92916

# run permutation
if 'seed' in globals():
    rng_permute = np.random.RandomState(seed)
    idx_permute = rng_permute.permutation(n_steps)
else:
    idx_permute = np.random.permutation(n_steps)

# pre-loading
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# build computation graph
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, n_steps, input_dims])
y = tf.placeholder(tf.float32, [None, n_classes])    
global_step = tf.Variable(0, name='global_step', trainable=False)

# build prediction graph
my_logger.info("==> Building a dRNN with %s cells" % cell_type)
pred = drnn_classification(x, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)

# build loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# evaluation model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True #在昇腾AI处理器执行测试
custom_op.parameter_map["mix_compile_mode"].b = False  #关闭混合计算，根据实际情况配置，默认关闭
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
sess = tf.Session(config=config)

sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, './checkpoints_npu/' + cell_type + '/best_model.ckpt')

step = 0
batch_size = 128

testing_iters = batch_size * 2

while step * batch_size < testing_iters:
    # test performance
    batch_x = mnist.test.images
    batch_y = mnist.test.labels
    batch_x = batch_x[:, idx_permute]        
    batch_x = batch_x.reshape([-1, n_steps, input_dims])
    feed_dict = {
        x : batch_x,
        y : batch_y
    }
    cost_, accuracy_, step_ = sess.run([cost, accuracy, global_step], feed_dict=feed_dict)
    test_results.append((step_, cost_, accuracy_))
    
    my_logger.info("========> Testing Accuarcy: " + "{:.6f}".format(accuracy_))
    
    step += 1