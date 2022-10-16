# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
from npu_bridge.estimator import npu_ops
from tensorflow.python.framework import graph_util
#from ops import *
#from data import *
#from net import *
#from utils import *
# 添加导入NPU库的头文件
import npu_bridge
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from absl import flags
from tensorflow.contrib import training as contrib_training
from generative import  SeqModel
from generative import load_datasets
from generative import create_out_dir

# parameters
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('ckpt_path', './log/ckpt_file/model_218000.ckpt',
                    'directory of ckpt')
FLAGS = tf.app.flags.FLAGS
config = tf.ConfigProto()
# keras迁移添加内容
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap


#ckpt_path = "./log/ckpt_file/model_218000.ckpt"


def main():
    tf.reset_default_graph()
    ckpt_path = FLAGS.ckpt_path
    params = contrib_training.HParams(
        num_steps=FLAGS.num_steps,
        val_freq=FLAGS.val_freq,
        seq_len=FLAGS.seq_len,
        batch_size=FLAGS.batch_size,
        emb_variable=FLAGS.emb_variable,
        emb_size=FLAGS.emb_size,
        vocab_size=4,
        hidden_lstm_size=FLAGS.hidden_lstm_size,
        norm_lstm=FLAGS.norm_lstm,
        dropout_rate=FLAGS.dropout_rate,
        learning_rate=FLAGS.learning_rate,
        reg_type=FLAGS.reg_type,
        reg_weight=FLAGS.reg_weight,
        out_dir=FLAGS.out_dir,
        in_tr_data_dir=FLAGS.in_tr_data_dir,
        in_val_data_dir=FLAGS.in_val_data_dir,
        ood_val_data_dir=FLAGS.ood_val_data_dir,
        master=FLAGS.master,
        save_meta=FLAGS.save_meta,
        filter_label=FLAGS.filter_label,
        mutation_rate=FLAGS.mutation_rate,
    )
    create_out_dir(params)
    x_in =tf.placeholder(tf.int32,shape=[100,250],name="x_in")
    y_in =tf.placeholder(tf.int32,shape=[100,],name="y_in") 
    model = SeqModel(params,x_in,y_in)
    model.reset()
    loss_i = tf.identity(model.loss_i, name='loss_i')
    with tf.Session(config=config) as sess:
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, './log/frozen_pb_file', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./log/frozen_pb_file/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='loss_i',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./log/pb_file/result.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()
