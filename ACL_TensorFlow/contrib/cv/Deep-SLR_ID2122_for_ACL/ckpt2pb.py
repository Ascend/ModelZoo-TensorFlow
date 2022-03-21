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
This is a demo for the conversion of a training checkpoint to a pb model ON MODELARTS.
To use it, you need to train your model on NPU in the first place, and run this script on ModelArts.
You have to modify it for other use according to your actual demands.
"""

import tensorflow as tf
from npu_bridge.npu_init import RewriterConfig, npu_config_proto, ExponentialUpdateLossScaleManager, \
    NPULossScaleOptimizer
from tensorflow.python.tools import freeze_graph
import HDSLR as mm
import getopt
import sys
import os

opts, args = getopt.gnu_getopt(sys.argv[1:], 'd:o:s:', ['data_path=', 'output_path=', 'steps='])
data_path = opts[0][1]
output_path = opts[1][1]

K = 1

tf.reset_default_graph()

# Define the input nodes of the network. Keep the size of input the same as the one from training
atbT = tf.placeholder(tf.float32, shape=(1, 24, 256, 232, 1), name='atb')
maskT = tf.placeholder(tf.complex64, shape=(1, 12, 256, 232, 1), name='mask')

# Call the model maker to produce a model for inference
outk = mm.makeModel(atbT, maskT, K)
fhatT = outk['dc' + str(K)]
fhatT = tf.identity(fhatT, name='fhat')  # Get the output node a determined name!

# Modify the input directory of your checkpoint
input_ckpt = os.path.join(data_path, '20Nov_1026_360I_100E_1B_1K/model-100')

with tf.Session() as sess:
    # Save the graph and produce a 'tmp_model.pb' file at 'output_path', which is passed to the freeze_graph function
    # as the input_graph.
    tf.io.write_graph(sess.graph_def, output_path, 'tmp_model.pb')  # Produce a model file through write_graph
    freeze_graph.freeze_graph(
        input_graph=os.path.join(output_path, 'tmp_model.pb'),  # Modify the input directory of the model file
        input_saver='',
        input_binary=False,
        input_checkpoint=input_ckpt,  # Modify the input checkpoint file prefix
        output_node_names='fhat',  # Keep it the same with the name of the output node above!
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(output_path, 'deep-slr-model-100.pb'),  # Modify the name of the output pb file
        clear_devices=False,
        initializer_nodes='')
