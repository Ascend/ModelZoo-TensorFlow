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
"""Transform checkpoint file to pb model."""
import os
import tensorflow as tf
# from tensorflow_core.python.tools import freeze_graph
import freeze_graph
from mixnet import mixnet_builder


def ckpt2pb():
    """Transform checkpoint file to pb model."""

    CKPT_PATH = "./ckpt/model.ckpt-300000"
    DST_FOLDER = './pb_model'

    tf.compat.v1.reset_default_graph()
    # 定义网络的输入节点，输入大小与模型在线测试时一致
    inputs = tf.compat.v1.placeholder(
        tf.float32, shape=[1, 224, 224, 3], name="input")

    logits, _ = mixnet_builder.build_model(
        inputs,
        model_name="mixnet-s",
        training=False)  # 获得模型的输出节点，此处为最后一个conv的输出，未经过softmax，inference时如果只需要index可以不使用
    print(logits)

    # 重新定义网络的输出节点
    # 重新定义一个identity节点name方便后续freeze_graph时指向；因为后续离线推理时候我额外使用了tf.argmax，所以此处未定义。
    logits = tf.identity(logits, name='output')

    with tf.compat.v1.Session() as sess:
        # 保存图，在 DST_FOLDER 文件夹中生成tmp_model.pb文件
        # tmp_model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.io.write_graph(sess.graph_def, DST_FOLDER,
                          'model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            # 传入write_graph生成的模型文件
            input_graph=os.path.join(DST_FOLDER, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=CKPT_PATH,  # 传入训练生成的checkpoint文件
            output_node_names='output',  # 与重新定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(
                DST_FOLDER, 'mixnet.pb'),   # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')


if __name__ == '__main__':
    ckpt2pb()
