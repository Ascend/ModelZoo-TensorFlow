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
# from create_tf_record import *
from tensorflow.python.framework import graph_util


def freeze_graph(input_checkpoint ,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB
    :return:
    '''

    output_node_names = "Model/generator/B_t"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 
    input_graph_def = graph.as_graph_def()  # 

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 
        output_graph_def = graph_util.convert_variables_to_constants(  # ，
            sess=sess,
            input_graph_def=input_graph_def  ,# :sess.graph_def
            output_node_names=output_node_names.split(",")  )# ，

        with tf.gfile.GFile(output_graph, "wb") as f:  # 
            f.write(output_graph_def.SerializeToString())  # 
        print("%d ops in the final graph." % len(output_graph_def.node))  # 

# input_checkpoint='inceptionv1/model.ckpt-0'
# out_pb_path='inceptionv1/frozen_model.pb'

input_checkpoint ='C:/Users/xkh/Desktop/beautyGAN-100'
out_pb_path ='./results/frozen_model.pb'
freeze_graph(input_checkpoint, out_pb_path)

