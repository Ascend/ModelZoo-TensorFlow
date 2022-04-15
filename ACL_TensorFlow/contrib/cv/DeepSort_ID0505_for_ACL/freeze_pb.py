# Copyright 2020 Huawei Technologies Co., Ltd
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

base_path=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/../")

from nets.deep_sort.network_definition import create_network


def main():

    tf.reset_default_graph()

    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[1, 128, 64, 3], name="input")

    features, logits = create_network(inputs, \
                                      num_classes=1502, \
                                      add_logits=True, \
                                      reuse=None, \
                                      create_summaries=False, \
                                      weight_decay=1e-8)

    prediction = tf.argmax(input=logits, axis=-1, output_type=tf.dtypes.int32, name="output")

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph="deep_sort.pb"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, "/home/HwHiAiUser/deep/lognckpt_bak/KOOKKJ/model.ckpt-98077")

        output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["output"])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")

if __name__ == '__main__':
    main()

