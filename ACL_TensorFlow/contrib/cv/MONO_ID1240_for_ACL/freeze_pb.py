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
from net import Net
from tensorflow.python.framework import graph_util

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = tf.to_float(min_disp) + tf.to_float(max_disp - min_disp) * disp
    depth = tf.to_float(1.) / scaled_disp
    return depth


def main():
    tf.reset_default_graph()
    with tf.name_scope('data_loading'):
        input_node_uint8 = tf.placeholder(tf.uint8, shape=(1, 192, 640, 3))
        input_node = tf.image.convert_image_dtype(input_node_uint8, dtype = tf.float32)
        input_node = (input_node - 0.45) / 0.225
    with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
        net_builder = Net(False)
        res18_tc, skips_tc = net_builder.build_resnet18(input_node)
        pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)
        pred_disp_rawscale = [tf.image.resize_bilinear(pred_disp[i], [192, 640]) for i in range(4)]
        pred_depth_rawscale = disp_to_depth(pred_disp_rawscale, 0.1, 100)
        pred_depth = pred_depth_rawscale[0]#192 640 4 8

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph = "mono2_strided.pb"
    #print(pred_depth)
    #tf.reset_default_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "/home/smartcar/Downloads/Convert_pb/checkpoints/model-402482")

        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['monodepth2_model/strided_slice'])

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")


if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        print(e)
