# ============================================================================
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
import config
import argparse
import dnnlib.tflib as tflib
import os


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptdir', type=str, default=r'model/ckpt_npu/model.ckpt', help='ckpt files input path .')
    parser.add_argument('--pbdir', type=str, default=r'model/pb', help='pb files output path.')
    parser.add_argument('--input-node-name', type=str, default=r'input', help='input node names.')
    parser.add_argument('--output-node-name', type=str, default=r'output', help='output node names.')
    parser.add_argument('--width', type=int, default=768, help='input node width')
    parser.add_argument('--height', type=int, default=512, help='input node height')
    parser.add_argument('--pb-name', type=str, default=r'test.pb', help='pd file name.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    ckpt_path = args.ckptdir
    output_path = os.path.join(args.pbdir, args.pb_name)
    height = args.height
    width = args.width

    tf.reset_default_graph()

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, shape=[None, 3, height, width], name=args.input_node_name)
        
        net = tflib.Network(**config.net_config)
        logits = net.get_output_for(inputs, width=width, height=height)

        
        one = tf.constant(1, dtype=tf.float32)
        output = tf.multiply(logits, one, name=args.output_node_name)

        tf.train.write_graph(sess.graph_def, args.pbdir, args.pb_name)
        freeze_graph.freeze_graph(
            input_graph=output_path,
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names=args.output_node_name,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=output_path,
            clear_devices=False,
            initializer_nodes='')
        print("[info] Pb file:{}".format(output_path))


if __name__ == '__main__':
    main()
