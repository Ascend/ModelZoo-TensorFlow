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
import argparse
import os
import dan_model_pb


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptdir', type=str, default=r'model/ckpt/model.ckpt',
                        help='ckpt files input path .')
    parser.add_argument('--pbdir', type=str, default=r'model/pb/', help='pb files output path.')
    parser.add_argument('--input_node_name', type=str, default=r'input', help='input node names.')
    parser.add_argument('--output_node_name', type=str, default=r'output', help='output node names.')
    parser.add_argument('--width', type=int, default=112, help='input node width')
    parser.add_argument('--height', type=int, default=112, help='input node height')
    parser.add_argument('--pb_name', type=str, default=r'test.pb', help='pd file name.')
    parser.add_argument('--num_lmark', '-nlm', type=int, default=68)
    args = parser.parse_args()
    return args


class VGG16Model(dan_model_pb.Model):
    def __init__(self, num_lmark, data_format=None):
        img_size = 112
        filter_sizes = [64, 128, 256, 512]
        num_convs = 2
        kernel_size = 3

        super(VGG16Model, self).__init__(
            num_lmark=num_lmark,
            img_size=img_size,
            filter_sizes=filter_sizes,
            num_convs=num_convs,
            kernel_size=kernel_size,
            data_format=data_format
        )


def main():
    args = parse_args()

    ckpt_path = args.ckptdir
    output_path = os.path.join(args.pbdir, args.pb_name)
    height = args.height
    width = args.width

    tf.reset_default_graph()

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, shape=[1, height, width, 1], name=args.input_node_name)

        model = VGG16Model(args.num_lmark)
        logits = model(inputs_imgs=inputs, s1_training=False, s2_training=False, mean_shape=None,
                       imgs_mean=None, imgs_std=None)
        print(logits['s2_ret'])

        one = tf.constant(1, dtype=tf.float32)
        output = tf.multiply(logits['s2_ret'], one, name='output')
        # output = tf.reshape(logits['s2_ret'], (-1, 68, 2), name='output')

        tf.train.write_graph(sess.graph_def, args.pbdir, args.pb_name)

        # init = tf.compat.v1.global_variables_initializer()
        # sess.run(init)

        freeze_graph.freeze_graph(
            input_graph=output_path,
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=output_path,
            clear_devices=False,
            initializer_nodes='')
        print("[info] Pb file:{}".format(output_path))


if __name__ == '__main__':
    main()
