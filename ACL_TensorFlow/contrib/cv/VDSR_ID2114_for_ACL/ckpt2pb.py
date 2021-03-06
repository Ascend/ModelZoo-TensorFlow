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
import os, argparse
import moxing as mox
from google.protobuf import text_format

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/dataset-jcr/电子器件-分类/node/")   #obs该目录下放ckpt4个模型文件或pb文件或pbtxt文件
parser.add_argument("--train_url", type=str, default="/dataset-jcr/")
args = parser.parse_args()
data_dir = "/cache/dataset"
os.makedirs(data_dir)
mox.file.copy_parallel(args.data_url, data_dir)

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
    :param output_graph: PB模型保存路径
    :return:
    '''
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "shared_model/Add,Placeholder"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(
            # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开)
        with tf.io.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":
    # 下面这段用于把ckpt转成pb
    # 输入ckpt模型路径
    #input_checkpoint=os.path.join(args.data_url, "VDSR_adam_epoch_099.ckpt-336400")
    # 输出pb模型的路径
    #out_pb_path=os.path.join(args.data_url, "VDSR_model.pb") # 调用freeze_graph将ckpt转为pb
    #freeze_graph(input_checkpoint,out_pb_path)

    # 下面这段用于把pb转成txt，然后自己就可以手动修改模型结构，
    #with tf.gfile.GFile(out_pb_path, "rb") as f:
        #graph_def = tf.GraphDef()
        #graph_def.ParseFromString(f.read())
        #tf.import_graph_def(graph_def, name='')
        #tf.train.write_graph(graph_def, args.data_url, 'model.txt', as_text=True)

    # 下面这段用于把修改后的txt转成pb模型
    with tf.gfile.FastGFile(os.path.join(args.data_url, "model.txt"), 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, args.data_url, 'model.pb', as_text=False)

mox.file.copy_parallel(args.data_url, args.train_url)
