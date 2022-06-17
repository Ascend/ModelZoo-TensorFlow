# -*- coding: utf-8 -*-
#
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
#
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm

import npu_device

import argparse
import ast
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--data_path', default='./data/')
parser.add_argument('--model_path', default='./model/w2v_model')
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval,help='auto_tune flag, default is False')
parser.add_argument('--static', dest='static', type=ast.literal_eval, help='if or not use static shape, default is False')
args = parser.parse_args()
def npu_config():
    if args.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = args.data_dump_path
        npu_device.global_options().dump_config.dump_step = args.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if args.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = args.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if args.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = args.precision_mode
    if args.use_mixlist and args.precision_mode=='allow_mix_precision':
        npu_device.global_options().modify_mixlist=args.mixlist_file
    if args.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=args.fusion_off_file
    if args.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()
#===============================NPU Migration=========================================
npu_config()
# [train_classifier, interactive_predict, train_word2vec, save_model, test]
mode = 'train_classifier'

word2vec_config = {
    'stop_words': '%s/data/w2v_data/stop_words.txt' % args.data_path,  # 停用词(可为空)
    'train_data': '%s/data/w2v_data/comments_data.csv' % args.data_path,  # 词向量训练用的数据
    'model_dir': args.model_path,  # 词向量模型的保存文件夹
    'model_name': 'w2v_model.pkl',  # 词向量模型名
    'word2vec_dim': 300,  # 词向量维度
    'precision_mode': args.precision_mode,
    'over_dump':args.over_dump,
    'data_dump_flag':args.data_dump_flag,
    'data_dump_step':args.data_dump_step,
    'profiling':args.profiling,
    'profiling_dump_path':args.profiling_dump_path,
    'over_dump_path':args.over_dump_path,
    'data_dump_path':args.data_dump_path,
    'use_mixlist':args.use_mixlist,
    'fusion_off_flag':args.fusion_off_flag,
    'mixlist_file':args.mixlist_file,
    'fusion_off_file':args.fusion_off_file,
    'auto_tune':args.auto_tune
}

#CUDA_VISIBLE_DEVICES = 0
# int, -1:CPU, [0,]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

classifier_config = {
    # 模型选择
    'classifier': 'textcnn',
    # 训练数据集
    'train_file': '%s/data/data/train_data.csv' % args.data_path,
    # 验证数据集
    'val_file': '%s/data/data/dev_data.csv' % args.data_path,
    # 测试数据集
    'test_file': '',
    # token粒度,token选择字粒度的时候，词嵌入无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'word',
    # 引入外部的词嵌入,可选word2vec、Bert
    # 此处只使用Bert Embedding,不对其做预训练
    # None:使用随机初始化的Embedding
    'embedding_method': 'word2vec',
    # 不外接词向量的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': '%s/token2id_word' % args.model_path,
    # 类别和对应的id
    'classes': {'negative': 0, 'positive': 1},
    # 模型保存的文件夹
    'checkpoints_dir': '%s/textcnn_word2vec' % args.model_path,
    # 模型保存的名字
    'checkpoint_name': 'textcnn_word2vec',
    # 卷集核的个数
    'num_filters': 64,
    # 学习率
    'learning_rate': 0.001,
    # 训练epoch
    'epoch': args.epochs,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 1,
    # 是否提前结束
    'is_early_stop': False,
    # 是否引入attention
    # 注意:textrcnn不支持
    'use_attention': False,
    # attention大小
    'attention_size': 300,
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    # 遗忘率
    'dropout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn中需要设定
    'hidden_dim': 200,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False,
    'static': args.static
}
