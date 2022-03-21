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
from tensorflow.python.tools import freeze_graph
import os
# from npu_bridge.npu_init import *
# 导入网络模型文件
# 如果使用modelnet10/40数据集进行转化则导入vsl网络模型
from vsl import VarShapeLearner
# # 如果使用pascal3D数据集进行转化则导入vsl_rec网络模型
# from vsl_rec import VarShapeLearner

# 指定checkpoint路径，请根据所选择的数据集选择自己ckpt路径
# modelnet10数据集ckpt路径
ckpt_path = "parameters/modelnet10-0499-cost-0.0068.ckpt"

# # modelnet40数据集ckpt路径
# ckpt_path = "parameters/modelnet40-0499-cost-0.0076.ckpt"
# # pascal3D数据集ckpt路径
# ckpt_path = "parameters/aero-008-3-2-5-cost-0.0045.ckpt"

# define network structure, parameters
# 以下参数需依据选取的数据集来进行选取，具体请参见README。
global_latent_dim  = 10
local_latent_dim   = 5
local_latent_num   = 5
obj_res     = 30
batch_size  = 100


def main():
    # load VSL model
    VSL = VarShapeLearner(obj_res=obj_res,
                          batch_size=batch_size,
                          global_latent_dim=global_latent_dim,
                          local_latent_dim=local_latent_dim,
                          local_latent_num=local_latent_num)

    # load saved parameters here, comment this to train model from scratch.
    VSL.saver.restore(VSL.sess, os.path.abspath(ckpt_path))

    input_shape = [batch_size, obj_res, obj_res, obj_res, 1]
    VSL.x = tf.placeholder(tf.float32, shape=input_shape, name='input')  # 输入节点

    latent_feature = tf.concat([VSL.z_mean[i] for i in range(local_latent_num + 1)], axis=1, name="latent_feature") # 输出节点

    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')  # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',  # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names='latent_feature',  # 与定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/modelnet10.pb',  # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()
