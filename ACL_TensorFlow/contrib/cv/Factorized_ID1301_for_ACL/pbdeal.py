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
from tensorflow.python.framework import graph_util
from google.protobuf import text_format



def convert_pbtxt_to_pb(filename):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()

        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './tmp/train', 'model.pb', as_text=False)
    return
def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB??????????????????
    :return:
    '''
    # ???????????????????????????,???????????????????????????????????????????????????
    output_node_names = "cnn_tower/layer_6/BatchNorm/Relu,cnn_tower_1/layer_6/BatchNorm/Relu"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #????????????????????????
        output_graph_def = graph_util.convert_variables_to_constants(
            # ????????????????????????????????????
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))# ?????????????????????????????????????????????)
        with tf.io.gfile.GFile(output_graph, "wb") as f: #????????????
            f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":

    #?????????????????????pb??????txt???????????????????????????????????????????????????
    with tf.gfile.GFile("new3.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, "./", 'model.txt', as_text=True)

    # ?????????????????????????????????txt??????pb??????
    # with tf.gfile.FastGFile(("./model.txt"), 'r') as f:
    #     graph_def = tf.GraphDef()
    #     file_content = f.read()
    #     #Merges the human-readable string in `file_content` into `graph_def`.
    #     text_format.Merge(file_content, graph_def)
    #     tf.train.write_graph(graph_def, "./", 'new.pb', as_text=False)