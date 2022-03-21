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
import model
import os
from tensorflow.python.tools import freeze_graph

batch_size=256
model_path = './ckpt/'
images = tf.placeholder(tf.float32, shape=[batch_size,28,28,1], name='input')
labels = tf.placeholder(tf.int64, [batch_size])
test_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_x')
test_labels = tf.placeholder(tf.int64, [None], name='input_y')
with tf.variable_scope('sphere20', reuse=tf.AUTO_REUSE):
    network = model.Model(images, labels)
with tf.variable_scope('sphere20', reuse=tf.AUTO_REUSE):
    test_network = model.Model(test_images, test_labels)
pred_prob = test_network.pred_prob
pred = tf.argmax(pred_prob,axis=1,name = 'output')
with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './pb_model', 'output_empty.pb')  # save pb file with output node
    freeze_graph.freeze_graph(
        input_graph='./pb_model/output_empty.pb',  # the pb file with output node
        input_saver='',
        input_binary=False,
        input_checkpoint=model_path + 'sphereface.ckpt',  # input checkpoint file path
        output_node_names='output',  # the name of output node in pb file
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./pb_model/sphereface.pb',  # path of output graph
        clear_devices=False,
        initializer_nodes='')
print('done')

