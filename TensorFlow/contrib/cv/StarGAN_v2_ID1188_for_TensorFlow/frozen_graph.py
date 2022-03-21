# -*- coding:utf-8 -*-
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
"""
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph
import os
import shutil
from tqdm import tqdm
import argparse

from StarGAN_v2 import StarGAN_v2
from utils import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *

"""parsing and configuration"""

def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def freeze(input_checkpoint, output_graph):
    '''

    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "generator/output"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    show_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, input_checkpoint)  #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        show_all_variables()

        with tf.gfile.GFile(output_graph, "wb") as f:  #保存模型
            f.write(output_graph_def.SerializeToString())  #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  #得到当前图有几个操作节点

        for op in sess.graph.get_operations():
            print(op.name, op.values())

    print("freeze_graph finished ...")

def frozen_graph(input_checkpoint, output_graph, args):
    '''

    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''

    # open session
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    with tf.Session(config=config) as sess:
        gan = StarGAN_v2(sess, args)

        # build graph
        gan.build_model()
        input_node1 = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name="input_node1")
        input_node2 = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name="input_node2")
        c = 1
        output_node = gan.generator(input_node1, tf.gather(gan.style_encoder(input_node2), c))
        output = tf.identity(output_node, name='the_outputs')
        saver = tf.train.Saver()
        saver.restore(sess, input_checkpoint)
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, input_checkpoint, 'the_outputs', 'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test or refer_test ?')
    parser.add_argument('--dataset', type=str, default='afhq-raw', help='dataset_name')
    parser.add_argument('--refer_img_path', type=str, default='refer_img.jpg', help='reference image path')

    parser.add_argument('--iteration', type=int, default=200000, help='The number of training iterations')

    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size') # each gpu
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=50000, help='decay start iteration')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')

    parser.add_argument('--adv_weight', type=float, default=1, help='The weight of Adversarial loss')
    parser.add_argument('--sty_weight', type=float, default=0.3, help='The weight of Style reconstruction loss') # 0.3 for animal
    parser.add_argument('--ds_weight', type=float, default=1, help='The weight of style diversification loss') # 1 for animal
    parser.add_argument('--cyc_weight', type=float, default=0.1, help='The weight of Cycle-consistency loss') # 0.1 for animal

    parser.add_argument('--r1_weight', type=float, default=1, help='The weight of R1 regularization')
    parser.add_argument('--gp_weight', type=float, default=10, help='The gradient penalty lambda')

    parser.add_argument('--gan_type', type=str, default='gan', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--ch', type=int, default=32, help='base channel number per layer')
    parser.add_argument('--n_layer', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    parser.add_argument('--style_dim', type=int, default=16, help='length of style code')

    parser.add_argument('--num_style', type=int, default=5, help='number of styles to sample')

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    parser.add_argument("--model_phase", default="frozen_graph", help="model phase: frozen_graph/test_pb.")
    parser.add_argument("--input_checkpoint", default="./checkpoint/model-200000", help="the path of checkpoint file.")
    parser.add_argument("--out_pb_path", default="./stargan_v2_model.pb", help="the path of pb file.")
    args = parser.parse_args()

    phase = args.model_phase
    input_checkpoint = args.input_checkpoint
    out_pb_path = args.out_pb_path

    # 生成pb模型
    if(phase == "frozen_graph"):
        frozen_graph(input_checkpoint, out_pb_path, args)

