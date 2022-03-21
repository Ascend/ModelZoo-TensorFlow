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
import importlib
import datetime
dt = datetime.datetime.now()
dt =dt.strftime("%m-%d_%H-%M")
dir = "./pb_model/" + dt
if not os.path.isdir(dir):
    os.makedirs(dir)
ckpt_path = "/home/ma-user/modelarts/user-job-dir/code/log/sphere64_casia_am_PFE/20211215-121426/ckpt-3000"


def main():
    config = importlib.import_module("config.sphere64_casia")
    tf.reset_default_graph()
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 112, 96, 3], name="input")
    network = importlib.import_module('.sphere_net_PFE', 'models')
    mu, conv_final = network.inference(inputs, config.embedding_size)
    uncertainty_module = importlib.import_module('.uncertainty_module', 'models')
    log_sigma_sq = uncertainty_module.inference(conv_final, config.embedding_size, phase_train=False, scope='UncertaintyModule')
    mu = tf.identity(mu, name='mu')
    sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, dir, 'pfe_input_graph'+dt+'.pb')
        freeze_graph.freeze_graph(
            input_graph=dir + '/pfe_input_graph'+dt+'.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names='mu,sigma_sq',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=dir + '/pb_pfe'+dt+'.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()
