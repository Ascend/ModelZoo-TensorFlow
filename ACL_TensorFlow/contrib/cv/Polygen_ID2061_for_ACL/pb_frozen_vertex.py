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

vertex_ckpt = "ckpt/vertex/model"

vertex_module_config = dict(
    decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8, layer_norm=True,
        num_layers=24,
        dropout_rate=0.4,
        re_zero=True,
        memory_efficient=True
    ),
    quantization_bits=8,
    class_conditional=True,
    max_num_input_verts=5000,
    use_discrete_embeddings=True,
)

tf.reset_default_graph()

vertex_model = modules.VertexModel(**vertex_module_config)

num_samples_min = 1
num_samples_batch = 8
max_num_vertices = 400
max_num_face_indices = 2000
top_p_vertex_model = 0.9
top_p_face_model = 0.9

vertex_model_context = {'class_label': tf.placeholder(tf.int32, shape=[None, ], name="v_input")}
vertex_samples = vertex_model.sample(
    num_samples_batch, context=vertex_model_context,
    max_sample_length=max_num_vertices, top_p=top_p_vertex_model,
    recenter_verts=True, only_return_complete=True)

vertices_mask = tf.expand_dims(vertex_samples['vertices_mask'], 2)
vertices_dict = tf.concat([vertex_samples['vertices'], vertices_mask], axis=-1, name='v_output')


with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './pb', 'vertex_model.pb')
    freeze_graph.freeze_graph(
        input_graph='./pb/vertex_model.pb',  # 传入write_graph生成的模型文件
        input_saver='',
        input_binary=False,
        input_checkpoint=vertex_ckpt,  # 传入训练生成的checkpoint文件
        output_node_names='v_output',  # 与定义的推理网络输出节点保持一致
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./pb/vertex_model.pb',  # 改为需要生成的推理网络的名称
        clear_devices=False,
        initializer_nodes='')
