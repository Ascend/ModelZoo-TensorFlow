"""LICENSE"""
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
import sys
import tensorflow as tf
from tensorboard.compat.proto.rewriter_config_pb2 import RewriterConfig

from Generator import Generator
from tensorflow_core.python.tools import freeze_graph
from cfg import make_config

##npu
flags = tf.app.flags
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpáu or cpu)")
flags.DEFINE_string("platform", "apulis", "Run on apulis/modelarts platform. Modelarts "
                                          "Platform has some extra data copy operations")
# The following params only useful on NPU chip mode
flags.DEFINE_boolean("npu_dump_data", False, "dump data for precision or not")
flags.DEFINE_boolean("npu_dump_graph", False, "dump graph or not")
flags.DEFINE_boolean("npu_profiling", False, "profiling for performance or not")
flags.DEFINE_boolean("npu_auto_tune", False, "auto tune or not. And you must set tune_bank_path param.")
FLAGS = flags.FLAGS
config = make_config(FLAGS)

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭


#数据集路径
data_path = sys.argv[1]
# 模型路径
#模型路径--图片路径
output_path = sys.argv[2]

CKPT_PATH = data_path + 'model/840/840.ckpt'
output_path = output_path + 'export_dir'

#CKPT_PATH = './model/280.ckpt'
#output_path = './export_dir'

tf.compat.v1.reset_default_graph()
# 定义网络的输入节点，输入大小与模型在线测试时一致
SOURCE_VIS = tf.compat.v1.placeholder(tf.float32, shape=[1, 268, 360, 1], name="SOURCE_VIS")
SOURCE_IR = tf.compat.v1.placeholder(tf.float32, shape=[1, 268, 360, 1], name="SOURCE_IR")

#SOURCE_VIS = tf.compat.v1.placeholder(tf.float32, shape=[1, 576, 768, 1], name="SOURCE_VIS")
#SOURCE_IR = tf.compat.v1.placeholder(tf.float32, shape=[1, 576, 768, 1], name="SOURCE_IR")
# 调用网络模型生成推理图，用法参考slim
G = Generator('Generator')
generated_img = G.transform(vis=SOURCE_VIS, ir=SOURCE_IR)
print('generate:', generated_img.shape)
output = tf.identity(generated_img, name='output')

with tf.compat.v1.Session(config=config) as sess:
#with tf.compat.v1.Session() as sess:
    #保存图，在 DST_FOLDER 文件夹中生成tmp_model.pb文件
    # tmp_model.pb文件将作为input_graph给到接下来的freeze_graph函数
    tf.io.write_graph(sess.graph_def, output_path, 'tmp_model.pb')    # 通过write_graph生成模型文件
    freeze_graph.freeze_graph(
            input_graph=os.path.join(output_path, 'tmp_model.pb'),   # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=CKPT_PATH,  # 传入训练生成的checkpoint文件
            output_node_names='output',  # 与重新定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(output_path, 'DDcGAN.pb'),   # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')