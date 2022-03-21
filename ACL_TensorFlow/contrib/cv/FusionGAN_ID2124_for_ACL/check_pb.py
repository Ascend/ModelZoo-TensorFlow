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

from tensorflow.python.platform import gfile
model_path='./fusionGAN8.pb' 

# read graph definition
f = gfile.FastGFile(model_path, "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())

# fix nodes
for node in graph_def.node:
    if node.op == 'RefSwitch':
      node.op = 'Switch'
      for index in range(len(node.input)):
        if 'moving_' in node.input[index]:
          node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
      node.op = 'Sub'
      if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
      node.op = 'Add'
      if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'Assign':
      node.op = 'Identity'
      if 'use_locking' in node.attr: del node.attr['use_locking']
      if 'validate_shape' in node.attr: del node.attr['validate_shape']
      if len(node.input) == 2:
        # input0: ref: Should be from a Variable node. May be uninitialized.
        # input1: value: The value to be assigned to the variable.
        node.input[0] = node.input[1]
        del node.input[1]

# import graph into session
tf.import_graph_def(graph_def, name='')
tf.train.write_graph(graph_def, './', 'good_frozen8.pb', as_text=False)
tf.train.write_graph(graph_def, './', 'good_frozen8.pbtxt', as_text=True)
