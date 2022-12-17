##  Copyright 2021 Huawei Technologies Co., Ltd
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
from multiprocessing.heap import rebuild_arena
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
from tensorflow.python.framework import graph_util
from ops import *
from data import *
from utils import *
from datasets import *
import rebar 
import rebar_train
import numpy as np
import datasets 
#添加导入NPU库的头文件
import npu_bridge 
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from rebar_train import eval


config = tf.ConfigProto()
# keras迁移添加内容
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap

batch_size = 12

#dataset = DataSet("./train.txt", 30, batch_size)

ckpt_path = "./root/rebar/data/output.trial_1/model.ckpt-2000847"

def main():
    tf.reset_default_graph()
    inputs1 = tf.placeholder(tf.float32, [None,1,784 ], name="input1")
    FLAGS = tf.flags.FLAGS
    hparams = rebar.default_hparams
    hparams.parse(FLAGS.hparams)
    print(hparams.values())
    train_xs, valid_xs, test_xs = datasets.load_data(hparams)
    print(train_xs.shape)
    mean_xs = np.mean(train_xs, axis=0)  # Compute mean centering on training

    #training_steps = 2000000
    model = getattr(rebar, hparams.model)
    sbn = model(hparams, mean_xs)
    n_samples = 100
    

    #c_logits = net.conditioning_logits
    #p_logits = net.prior_logits
    #c_logits1, c_logits2 = tf.identity(sbn._create_train_op, name='c_logits1, c_logits2')
    with tf.Session(config=config) as sess:
        sbn.initialize(sess)
        batch_xs = valid_xs[0:24]
        #c_logits = sbn.partial_eval(batch_xs, n_samples)
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, './pb_model1021', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='Mean_11',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model1021/rebar1021.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()

'''


def freeze_graph(input_checkpoint,output_graph):

    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = "cond_1/Merge"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

# input_checkpoint='inceptionv1/model.ckpt-0'
# out_pb_path='inceptionv1/frozen_model.pb'

input_checkpoint='./output/model.ckpt-280000'
out_pb_path='frozen_model.pb'
freeze_graph(input_checkpoint, out_pb_path)
'''