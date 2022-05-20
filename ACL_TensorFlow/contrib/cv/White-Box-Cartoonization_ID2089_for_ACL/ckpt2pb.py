# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os, sys
import argparse
from npu_bridge.estimator import npu_ops

base_path=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/../")

import network
import guided_filter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default="", help="""set checkpoint path""")
    parser.add_argument("--patch_size", default=256, type=int)
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def main():
    args = parse_args()

    tf.reset_default_graph()

    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, args.patch_size, args.patch_size, 3], name="input")

    output = network.unet_generator(inputs)
    final_out = guided_filter.guided_filter(inputs, output, r=1, eps=5e-3)
    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph="wbcnet.pb"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(var_list=gene_vars)
        saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
        output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["add_1"])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")

if __name__ == '__main__':
    main()
