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
# ============================================================================

"""
Here, we show two methods to generate *.pb file.

An example of command-line usage is:
python3.7 frozen_graph.py --ckpt_path=./checkpoint/log
"""
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import argparse
from model import prototypical, euclidean_distance

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', default="./checkpoint/model.ckpt", help="set checkpoint path required=True,")
    # parser.add_argument('--num_classes', default=5, help="number of classes for datasets")
    parser.add_argument('--graph_dir', default="./checkpoint/log", help="set graph directory")
    parser.add_argument('--pb_name', default="proto_model.pb", help="set pb file name")
    # parser.add_argument('--chip', default="npu", help="Run on which chip, (npu or gpu or cpu)")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
        
    return args

## Using API 'convert_variables_to_constants', but the function will removed in future.
## https://www.tensorflow.org/api_docs/python/tf/compat/v1/graph_util/convert_variables_to_constants
def convert_variables_to_constants(sess, args):
    saver = tf.compat.v1.train.Saver()    
    saver.restore(sess, args.ckpt_path)

    output_node_names = ['output'] ## output node names must be a list
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names
    )
             
    ## Finally we serialize and dump the output graph to the filesystem
    output_graph_path = "%s/%s" % (args.graph_dir, args.pb_name)
    with tf.io.gfile.GFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_model_graph(sess, args, graph_name = "graph.pbtxt"):
    ## # save unfrozen graph, write graph proto to a file
    ## links: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/io/write_graph
    # fix batch norm nodes
    write_graph_def = sess.graph_def
    for node in  write_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    tf.io.write_graph(write_graph_def, args.graph_dir, graph_name)
    input_graph_path = "%s/%s" % (args.graph_dir, graph_name)
    output_graph_path = "%s/%s" % (args.graph_dir, args.pb_name)

    ## refer to links as follow
    ## https://zhuanlan.zhihu.com/p/64099452
    ## https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
    ## https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py
    freeze_graph.freeze_graph(
            input_graph=input_graph_path,
            input_saver='',
            input_binary=False, 
            input_checkpoint=args.ckpt_path,
            output_node_names='output',             # graph outputs node
            restore_op_name='save/restore_all',      # defautl value. unused in future.
            filename_tensor_name='save/Const:0',     # defautl value. unused in future. 
            output_graph=output_graph_path,          # graph outputs path
            clear_devices=False,
            initializer_nodes='')

def main():
    args = parse_args()
    tf.reset_default_graph()
    if not tf.gfile.Exists(args.graph_dir):
        tf.gfile.MakeDirs(args.graph_dir)

    im_height = 28
    im_width = 28
    channels = 1
    h_dim = 64
    z_dim = 64

    # set inputs node, here you should use placeholder.
    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels], name="inputx")
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels], name="inputq")
    y = tf.placeholder(tf.int64, [None, None], name="inputy")
    # process the inputs node.
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    num_classes, num_support = x_shape[0], x_shape[1]
    num_queries = q_shape[1]
    y_one_hot = tf.one_hot(y, depth=num_classes)

    # create inference graph
    emb_x = prototypical(x=tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim=h_dim,
                         z_dim=z_dim, reuse=False)
    emb_dim = tf.shape(emb_x)[-1]
    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)

    emb_q = prototypical(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim=h_dim,
                         z_dim=z_dim, reuse=True)
    dists = euclidean_distance(emb_q, emb_x)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(log_p_y, axis=-1), y), dtype=tf.float32), name="output")

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    with tf.Session(config=config) as sess:
        convert_variables_to_constants(sess, args)
        freeze_model_graph(sess, args)

    print("Done")

if __name__ == '__main__': 
    main()