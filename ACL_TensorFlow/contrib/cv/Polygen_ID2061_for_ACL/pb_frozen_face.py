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
from tensorflow.python.tools import freeze_graph
import modules

face_ckpt = "ckpt/face/model"

face_module_config = dict(
    encoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=10,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
    ),
    decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=14,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
    ),
    class_conditional=False,
    decoder_cross_attention=True,
    use_discrete_vertex_embeddings=True,
    max_seq_length=8000,
)

tf.reset_default_graph()

face_model = modules.FaceModel(**face_module_config)

num_samples_min = 1
max_num_vertices = 400
max_num_face_indices = 2000
top_p_vertex_model = 0.9
top_p_face_model = 0.9

vertex_samples = {'vertices': tf.placeholder(tf.float32, shape=[None, None, 3], name='f_input'),
                  'vertices_mask': tf.placeholder(tf.float32, shape=[None, None], name='f_mask')}
face_samples = face_model.sample(
    vertex_samples, max_sample_length=max_num_face_indices,
    top_p=top_p_face_model, only_return_complete=True)
faces = tf.tile(face_samples['faces'], [1, 1], name='f_output')

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './pb', 'face_model.pb')
    freeze_graph.freeze_graph(
        input_graph='./pb/face_model.pb',  # 传入write_graph生成的模型文件
        input_saver='',
        input_binary=False,
        input_checkpoint=face_ckpt,  # 传入训练生成的checkpoint文件
        output_node_names='f_output',  # 与定义的推理网络输出节点保持一致
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./pb/face_model.pb',  # 改为需要生成的推理网络的名称
        clear_devices=False,
        initializer_nodes='')
