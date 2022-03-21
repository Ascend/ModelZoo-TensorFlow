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
import os
from tensorflow.python.tools import freeze_graph
from dilated_rnn import drnn_classification

ckpt_path='./checkpoints_npu/GRU/best_model.ckpt'

# freeze graph while unknowing the output node's name
tf.reset_default_graph()

cell_type = 'GRU'
assert(cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 9
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
assert(len(hidden_structs) == len(dilations))
n_steps = 28 * 28
input_dims = 1
n_classes = 10 

input_node = tf.placeholder(tf.float32, [None, n_steps, input_dims])
output_node = drnn_classification(input_node, hidden_structs, dilations, n_steps, n_classes, input_dims, cell_type)

flow = tf.cast(output_node, tf.float16, 'the_outputs')
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, ckpt_path)
    tf.train.write_graph(sess.graph_def, './checkpoints_npu/GRU/pb', 'model.pb')
    freeze_graph.freeze_graph('./checkpoints_npu/GRU/pb/model.pb', '', False, ckpt_path, 'the_outputs',
                              'save/restore_all', 'save/Const:0', './checkpoints_npu/GRU/pb/gru.pb', False, "")