# coding=utf-8
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

from tensorflow.python.tools import freeze_graph
import tensorflow as tf

from maml import MAML


def run(ckpt_path):
    tf.reset_default_graph()

    inputa = tf.placeholder(tf.float32, shape=(4, 5, 1), name="inputa")
    inputb = tf.placeholder(tf.float32, shape=(4, 5, 1), name="inputb")
    labela = tf.placeholder(tf.float32, shape=(4, 5, 1), name="inputc")
    labelb = tf.placeholder(tf.float32, shape=(4, 5, 1), name="inputd")
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MAML(dim_input=1, dim_output=1, test_num_updates=1)
    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    logits = model.metaval_total_loss1
    tf.identity(logits, name="output")

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'output.pb')  # save pb file with output node
        freeze_graph.freeze_graph(
            input_graph='./pb_model/output.pb',  # the pb file with output node
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # input checkpoint file path
            output_node_names='output',  # the name of output node in pb file
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/mt-net.pb',  # path of output graph
            clear_devices=False,
            initializer_nodes='')


if __name__ == "__main__":
    ckpt_path = "./model59999"
    run(ckpt_path)
