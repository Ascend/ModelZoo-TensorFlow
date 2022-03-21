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
from tensorflow.python.tools import freeze_graph

import keras
from keras import backend as K
from keras.models import load_model, Model
import os
import os.path as osp
import numpy as np


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
  from tensorflow.python.framework.graph_util import convert_variables_to_constants
  graph = session.graph
  with graph.as_default():
      freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
      output_names = output_names or []
      output_names += [v.op.name for v in tf.global_variables()]
      input_graph_def = graph.as_graph_def()
      if clear_devices:
          for node in input_graph_def.node:
              node.device = ""
      frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                       output_names, freeze_var_names)
      return frozen_graph
input_fld = '/home/data/dataset/wxm/'
weight_file = 'model.ckpt'
output_graph_name = 'model_test.pb'

output_fld = input_fld + '/tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)

net = load_model(weight_file_path)
net = Model(input=net.input, output=net.get_layer('avg_pool').output)
net.build(input_shape=(224,224,3))

sess = K.get_session()

frozen_graph = freeze_session(K.get_session(), output_names=['avg_pool/AvgPool'])

from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))
  