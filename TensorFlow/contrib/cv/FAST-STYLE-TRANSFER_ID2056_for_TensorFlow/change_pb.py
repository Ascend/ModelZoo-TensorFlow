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
import os, sys
import argparse
from src.transform import net
import numpy as np

base_path=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/../")
#from model.unet3d import Builder
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default=1,
                        help="""set checkpoint path""")
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
    inputs = tf.placeholder(tf.float32, shape=[1, 256,256,3], name="input")
    logits = net(inputs)
    predition = tf.identity(logits, name="output")

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph="fast_style_transfer.pb"

    #with tf.Session() as sess:
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, args.ckpt_path)

        output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["output"])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")

if __name__ == '__main__':
    main()