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
from network import network
from tensorflow.python.tools import freeze_graph
import numpy as np

model_path = '/home/test_user05/pridnet/checkpoint'   

in_image = tf.placeholder(tf.float32, [None, None, None, 1],name='input')
output = network(in_image)
output = tf.minimum(tf.maximum(output, 0), 1,name='output')

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
    freeze_graph.freeze_graph(
        input_graph='./pb_model/model.pb',
        input_saver='',
        input_binary=False,
        input_checkpoint=model_path + '/model-3990.ckpt', 
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./pb_model/pridnet.pb',              
        clear_devices=False,
        initializer_nodes='')
print('done')
