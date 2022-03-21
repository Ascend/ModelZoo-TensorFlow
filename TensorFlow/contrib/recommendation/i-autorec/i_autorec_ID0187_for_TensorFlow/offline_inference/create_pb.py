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
# Copyright 2020 Huawei Technologies Co., Ltd
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
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.tools import freeze_graph

ckpt_path = 'pre_trained/ml-1m/model.ckpt'
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main():
    tf.reset_default_graph()

    num_items = 3952
    hidden_neuron = 1024

    input_R = tf.placeholder(dtype=tf.float32, shape=[None, num_items], name="input_R")
    input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, num_items], name="input_mask_R")

    V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[num_items, hidden_neuron],
                                         mean=0, stddev=0.03),dtype=tf.float32)
    W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[hidden_neuron, num_items],
                                         mean=0, stddev=0.03),dtype=tf.float32)
    mu = tf.get_variable(name="mu", initializer=tf.zeros(shape=hidden_neuron),dtype=tf.float32)
    b = tf.get_variable(name="b", initializer=tf.zeros(shape=num_items), dtype=tf.float32)

    pre_Encoder = tf.matmul(input_R, V) + mu
    Encoder = tf.nn.sigmoid(pre_Encoder)
    pre_Decoder = tf.matmul(Encoder, W) + b
    Decoder = tf.identity(pre_Decoder, name='output')

    with tf.Session(config=config) as sess:
        tf.train.write_graph(sess.graph_def, 'save', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='save/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='save/autorec.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

if __name__ == '__main__':
    main()

