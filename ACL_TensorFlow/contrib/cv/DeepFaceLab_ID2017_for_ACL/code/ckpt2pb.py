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

import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_checkpoint', required=True, type=str, \
                   help='input ckpt model')
    p.add_argument('-o', '--output_graph', required=True, type=str, \
                   help='folder of input image or file of other input.')
    p.add_argument('-n', '--output_nodes_name', required=True, choices=['Sigmoid_3', 'Sigmoid_4', 'Sigmoid_5'], \
                    help='Choose an output node name to specify the output pb model')
    return p.parse_args()


def freeze_graph(input_checkpoint, output_graph, output_node):

    output_node_names = output_node
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def main():
    args = get_args()
    input_checkpoint = args.input_checkpoint
    output_graph = args.output_graph
    output_node = args.output_nodes_name
    freeze_graph(input_checkpoint, output_graph, output_node)
    print('*'*20)
    print("[Info] Transfer ckpt model to pb graph successfully!")
    print('*'*20)


if __name__ == '__main__':
    main()
