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
from tensorflow.python.framework import graph_util
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--gen_num", type=int, default=5000, help="number of generated images")
    parser.add_argument("--output", type=str, default="../output", help="output path")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_h", type=int, default=32, help="image height")
    parser.add_argument("--img_w", type=int, default=32, help="image width")
    parser.add_argument("--train_img_size", type=int, default=32,
                        help="image will be resized to this size when training")
    # model arguments
    parser.add_argument("--base_channel", type=int, default=96, help="base channel number for G and D")
    parser.add_argument("--z_dim", type=int, default=120, help="latent space dimensionality")
    parser.add_argument("--ema", type=bool, default=False, help="use exponential moving average for G")
    parser.add_argument("--shared_dim", type=int, default=128, help="shared embedding dimensionality")
    args = parser.parse_args()

    # use different architectures for different image sizes
    if args.train_img_size == 128:
        from networks_128 import Generator, Discriminator
    elif args.train_img_size == 64:
        from networks_64 import Generator, Discriminator
    elif args.train_img_size == 32:
        from networks_32 import Generator, Discriminator

    # model path
    base_path = os.path.join(args.output, "model", str(args.train_img_size))
    model_path = os.path.join(base_path, "model.ckpt")
    ema_model_path = os.path.join(base_path, "ema.ckpt")
    ckpt_path = ema_model_path if args.ema else model_path

    # pb path
    pb_path = os.path.join(args.output, "pb_model", str(args.train_img_size))
    graph_pb_path = os.path.join(pb_path, "tmp_model.pb")
    model_pb_path = os.path.join(pb_path, "model.pb")
    final_pb_path = os.path.join(pb_path, "final_model.pb")

    tf.reset_default_graph()
    train_phase = tf.Variable(tf.constant(False, dtype=tf.bool), name="train_phase")
    # train_phase = tf.placeholder(tf.bool)                           # is training or not
    z = tf.placeholder(tf.float32, [None, args.z_dim], name="z")              # latent vector
    y = tf.placeholder(tf.int32, [None, 1], name="y")                            # class info
    y = tf.reshape(y, [-1])

    G = Generator("generator", args.base_channel)
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        embed_w = tf.get_variable("embed_w", [args.num_classes, args.shared_dim], initializer=tf.orthogonal_initializer())

    fake_img = G(z, train_phase, y, embed_w, args.num_classes)
    output = tf.identity(fake_img, name="output")

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, pb_path, "tmp_model.pb")
        # freeze model
        freeze_graph.freeze_graph(
            input_graph=graph_pb_path,
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names="output",
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=model_pb_path,
            clear_devices=False,
            initializer_nodes='')

    # see https://blog.csdn.net/u011765925/article/details/103038349 and
    # https://github.com/onnx/tensorflow-onnx/issues/77
    tf.reset_default_graph()
    with tf.gfile.FastGFile(model_pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
                if 'validate_shape' in node.attr:
                    del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
        with tf.Session() as sess:
            converted_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
        tf.train.write_graph(converted_graph_def, pb_path, "final_model.pb", as_text=False)

