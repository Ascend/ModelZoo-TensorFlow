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
# ============================================================================
"""
Here, we show two methods to generate *.pb file.

An example of command-line usage is:
python frozen_graph.py --ckpt_path=./log
"""
from npu_bridge.npu_init import *

import os
from copy import deepcopy
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import argparse
from src.flownet2 import FlowNet2
from src.training_schedules import LONG_SCHEDULE
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ckpt', type=str, default='./log/flownet2/model-loss-2.113736.ckpt',
                    help='the path of checkpoint')
parser.add_argument('--graph_dir', default="./offline_infer", help="set graph directory")
parser.add_argument('--image_size', type=list, default=[448, 1024], help="size of input images")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for one GPU")
parser.add_argument('--pb_name', default="flownet_tf_gpu.pb", help="set pb file name")
parser.add_argument('--chip', default="npu", help="Run on which chip, (npu or gpu or cpu)")


def main():
    flownet = FlowNet2()
    args = parser.parse_args()
    tf.reset_default_graph()
    if not tf.gfile.Exists(args.graph_dir):
        tf.gfile.MakeDirs(args.graph_dir)

    w, h = args.image_size
    input_v = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 2, w, h, 3], name='input_a')
    #input_a = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, w, h, 3], name='input_a')
    #input_b = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, w, h, 3], name='input_b')

    # 指定checkpoint路径
    inputs = {
            'input_a': input_v[:, 0, :],
            'input_b': input_v[:, 1, :],
        }
    pred_flow = flownet.model(inputs, LONG_SCHEDULE)
    pred_flow = pred_flow['flow']
    one = tf.constant(1, dtype=tf.float32)
    output = tf.multiply(pred_flow, one, name='output')

    # config = tf.ConfigProto()
    if args.chip == 'npu':
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.parameter_map['use_off_line'].b = True
        custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('allow_mix_precision')
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
            # 保存图，在./pb_model文件夹中生成model.pb文件
            # model.pb文件将作为input_graph给到接下来的freeze_graph函数
            tf.train.write_graph(sess.graph_def, './offline_infer/', 'model.pb')  # 通过write_graph生成模型文件
            freeze_graph.freeze_graph(
                input_graph='./offline_infer/model.pb',  # 传入write_graph生成的模型文件
                input_saver='',
                input_binary=False,
                input_checkpoint=args.ckpt,  # 传入训练生成的checkpoint文件
                output_node_names='output',  # 与定义的推理网络输出节点保持一致
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph='./offline_infer/flownet2.pb',  # 改为需要生成的推理网络的名称
                clear_devices=False,
                initializer_nodes='')
        print("done")
    
    else:
        with tf.Session(config=npu_config_proto()) as sess:
            # 保存图，在./pb_model文件夹中生成model.pb文件
            # model.pb文件将作为input_graph给到接下来的freeze_graph函数
            tf.train.write_graph(sess.graph_def, './offline_infer/', 'model.pb')  # 通过write_graph生成模型文件
            freeze_graph.freeze_graph(
                input_graph='./offline_infer/model.pb',  # 传入write_graph生成的模型文件
                input_saver='',
                input_binary=False,
                input_checkpoint=args.ckpt,  # 传入训练生成的checkpoint文件
                output_node_names='output',  # 与定义的推理网络输出节点保持一致
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph='./offline_infer/flownet2.pb',  # 改为需要生成的推理网络的名称
                clear_devices=False,
                initializer_nodes='')
        print("done")

if __name__ == '__main__':
    main()
