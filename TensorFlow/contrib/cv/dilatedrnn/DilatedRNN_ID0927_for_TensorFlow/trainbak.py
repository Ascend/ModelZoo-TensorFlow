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

# Training & Validation Process

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

# configurations
data_dir = "./dataset"
n_steps = 28 * 28
input_dims = 1
n_classes = 10 

parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='LSTM')
parser.add_argument('-epoch', type=int, default=30000)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-lr', type=float, default=1.0e-3)
opt = parser.parse_args()

# model config
cell_type = opt.type
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 9
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
assert(len(hidden_structs) == len(dilations))

# learning config
batch_size = opt.batch_size
learning_rate = opt.lr
training_iters = batch_size * opt.epoch
validation_step = 1000
display_step = 100

# permutation seed 
seed = 92916

# run permutation
mnist = input_data.read_data_sets(data_dir, one_hot=True)
if 'seed' in globals():
    rng_permute = np.random.RandomState(seed)
    idx_permute = rng_permute.permutation(n_steps)
else:
    idx_permute = np.random.permutation(n_steps)

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
optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)

# evaluation model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True # Train on NPU

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision") # Turn on mix precision
custom_op.parameter_map["mix_compile_mode"].b = False  # Turn off mix_mode
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # turn off remap
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF # turn off memory_optimization

sess = tf.Session(config=config)

sess.run(init)

step = 0
best_val_perf = 0.0
train_results = []
validation_results = []
test_results = []

saver = tf.train.Saver(max_to_keep=1)

my_logger.info("======== Start Training ========")
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)    
    batch_x = batch_x[:, idx_permute]
    batch_x = batch_x.reshape([batch_size, n_steps, input_dims])

    feed_dict = {
        x : batch_x, 
        y : batch_y
    }
    cost_, accuracy_, step_, _ = sess.run([cost, accuracy, global_step, optimizer], feed_dict=feed_dict)    
    train_results.append((step_, cost_, accuracy_))    

    if (step + 1) % display_step == 0:
        my_logger.info("Iter " + str(step + 1) + ", Minibatch Loss: " + "{:.6f}".format(cost_) \
        + ", Training Accuracy: " + "{:.6f}".format(accuracy_))

    step += 1

# validation performance
batch_x = mnist.validation.images
batch_y = mnist.validation.labels

# permute the data
batch_x = batch_x[:, idx_permute]        
batch_x = batch_x.reshape([-1, n_steps, input_dims])
feed_dict = {
    x : batch_x, 
    y : batch_y
}
cost_, accuracy__, step_ = sess.run([cost, accuracy, global_step], feed_dict=feed_dict)
validation_results.append((step_, cost_, accuracy__))
      
my_logger.info("========> Validation Accuarcy: " + "{:.6f}".format(accuracy__))

# store best validation performance model
if accuracy__ > best_val_perf:
    best_val_perf = accuracy__
    saver.save(sess, './checkpoints_npu/' + cell_type + '/best_model.ckpt')