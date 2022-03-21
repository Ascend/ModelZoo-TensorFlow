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


import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from devnet import *

def ckpt2pb2(model_name):
    """
    Convert ckpt to pb
    """
    tf.compat.v1.reset_default_graph()

    # 定义网络的输入节点，输入大小与模型在线测试时一致
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1440, 21])
    export_network = DeviationNetwork(inputs.shape, 4, learning_rate=0.001)
    export_network.load_weights(model_name)
    with tf.compat.v1.Session() as sess:
        #保存图，在 DST_FOLDER 文件夹中生成tmp_model.pb文件
        # tmp_model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.io.write_graph(sess.graph_def, "./model/pb/", 'tmp_model.pb', as_text=False)    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
		        input_graph="./model/pb/tmp_model.pb",   # 传入write_graph生成的模型文件
		        input_saver='',
		        input_binary=True, 
		        input_checkpoint=model_name,  # 传入训练生成的checkpoint文件
		        output_node_names="output/BiasAdd",  # 与重新定义的推理网络输出节点保持一致
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph=os.path.join( "./model/pb/", 'dev-net.pb'),   # 改为需要生成的推理网络的名称
		        clear_devices=False,
		        initializer_nodes='')

ckpt2pb2("./model/devnet_annthyroid_21feat_normalised_0.02cr_512bs_30ko_4d")