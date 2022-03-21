# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
model = "./resnest.ckpt-1"
output_node_names = "resize/ResizeBilinear"
out_pb_path = "./resnest.pb"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
with tf.device("/cpu:0"):
    reader = pywrap_tensorflow.NewCheckpointReader(model)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
    saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    for op in graph.get_operations():  # 打印出所有Graph中节点的名称。
        print(op.name)
    print("-----------------------------------------------------")
    #saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)

    saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = 	True
    with tf.Session(config=config) as sess:
        saver.restore(sess, model)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=[output_node_names])

        with tf.gfile.GFile(out_pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
