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
# ===========================
#   Author      : ChenZhou
#   Time        : 2021/11
#   Language    : Python
# ===========================

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
from masf_func import MASF
from tensorflow.python.platform import flags
import os
# 导入网络模型文件
tf.reset_default_graph()
FLAGS = flags.FLAGS

## Dataset PACS
flags.DEFINE_string('dataset', 'pacs', 'set the dataset of PACS')
flags.DEFINE_string('target_domain', 'art_painting', 'set the target domain')
flags.DEFINE_string('dataroot', '/home/ma-user/modelarts/user-job-dir/code/kfold/', 'Root folder where PACS dataset is stored')
flags.DEFINE_integer('num_classes', 7, 'number of classes used in classification.')

## Training options
flags.DEFINE_integer('train_iterations', 200, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 126, 'number of images sampled per source domain')
#zheli
flags.DEFINE_float('inner_lr', 1e-05, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 1e-05, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 1e-05, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/home/ma-user/modelarts/user-job-dir/code/log/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('summary_interval', 1, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 20, 'intervals to save model')
flags.DEFINE_integer('print_interval', 1, 'intervals to print out training info')
flags.DEFINE_integer('test_print_interval', 1, 'intervals to test the model')

class_list = {'0': 'dog',
              '1': 'elephant',
              '2': 'giraffe',
              '3': 'guitar',
              '4': 'horse',
              '5': 'house',
              '6': 'person'}


def main():
    """turn ckpt to pb"""
    ckpt_path = "/home/ma-user/modelarts/user-job-dir/code/itr152_model_acc0.5989558614143332"
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    tf.reset_default_graph()
    # 定义网络的输入节点
    model=MASF()
    input=tf.placeholder(tf.float32,shape=[None,227,227,3],name="input")
    # model.clip_value = FLAGS.gradients_clip_value
    # model.margin = FLAGS.margin
    model.KEEP_PROB = tf.placeholder(tf.float32)
    #
    model.weights = weights = model.construct_weights()
    #model.weights = weights = None
    #weights = get_weights()
    model.semantic_feature, outputs = model.forward_alexnet(input,weights)
    # 定义网络的输出节点
    acc = tf.identity(outputs, name='out')
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, '/home/ma-user/modelarts/user-job-dir/code/log/', 'model.pb')  # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            input_graph='/home/ma-user/modelarts/user-job-dir/code/log/model.pb',  # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names='out',  # 与定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='/home/ma-user/modelarts/user-job-dir/code/log/cartoon.pb',  # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()


