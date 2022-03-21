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

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import os

def freeze_graph(checkpoint_dir, pbtxt, output_graph):
    '''
    :param checkpoint_dir:
    :param output_graph: PB模型保存路径
    :return:
    '''

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "linear/output"
    saver = tf.train.import_meta_graph(pbtxt, clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    # for op in graph.get_operations():
    # print(op.name, op.values())


if __name__ == '__main__':
    checkpoint_dir = "./train_log/sparse-depth_100-k_24-c10-single-first150-second225-max300-0903-022838/"
    # pbtxt = os.path.join(checkpoint_dir, "graph-0903-022921.meta")
    pbtxt = os.path.join(checkpoint_dir, "graph.pbtxt")
    output_graph = os.path.join(checkpoint_dir, "model.pb")
    freeze_graph(checkpoint_dir, pbtxt, output_graph)
