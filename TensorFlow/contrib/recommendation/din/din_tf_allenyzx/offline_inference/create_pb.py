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
import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from din_model import DIN
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
from test_preprocess import pkl_load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--att_hidden_units', default='80,40')
parser.add_argument('--ffn_hidden_units', default='80,40')
parser.add_argument('--att_activation', default='sigmoid')
parser.add_argument('--ffn_activation', default='prelu')
parser.add_argument('--maxlen', default=40, type=int)
parser.add_argument('--input_checkpoint', default='')
parser.add_argument('--output_graph', default='')
parser.add_argument('--device', default='npu')


args = parser.parse_args()

if args.device == 'npu':
    from npu_bridge.npu_init import *

def main(input_checkpoint, output_graph, network):

    if args.device == 'npu':
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    else:
        config = tf.ConfigProto()


    tf.reset_default_graph()

    dense_inputs = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='dense_inputs')
    sparse_inputs = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='sparse_inputs')
    seq_inputs = tf.placeholder(shape=(None, 40, 1 + 1), dtype=tf.int32, name='seq_inputs')
    item_inputs = tf.placeholder(shape=(None, 1 + 1), dtype=tf.int32, name='item_inputs')

    flow = network([dense_inputs, sparse_inputs, seq_inputs, item_inputs])
    flow = tf.cast(flow, tf.float16, 'out')  # 设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, input_checkpoint)

        # save graph and freeze_graph
        tf.train.write_graph(sess.graph_def, os.path.dirname(input_checkpoint), 'temp.pb')
        freeze_graph.freeze_graph(input_graph=os.path.dirname(input_checkpoint) + '/temp.pb',
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=input_checkpoint,
                                  output_node_names='out',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph=output_graph,
                                  clear_devices=False,
                                  initializer_nodes="")

    print("create pb done")


if __name__ == '__main__':

    pkl = pkl_load('../raw_data/cache.pkl')

    network = DIN(att_hidden_units=[int(i) for i in args.att_hidden_units.split(',')],
                  ffn_hidden_units=[int(i) for i in args.ffn_hidden_units.split(',')],
                  att_activation=args.att_activation,
                  ffn_activation=args.ffn_activation,
                  maxlen=args.maxlen,
                  dnn_dropout=0,
                  feature_columns=pkl['feature_columns'],
                  behavior_feature_list=['item_id'],
                  device='cpu')
    main(args.input_checkpoint, args.output_graph, network)