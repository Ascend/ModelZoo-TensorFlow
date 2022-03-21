"""
to pb file
"""
# coding=utf-8
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

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import importlib
from lib import utils
#import moxing as mox
from networks import MobileFaceNet as mobilenet

class getinfo:
    def __init__(self):
        #self.model_path = "/home/lf/models/20211206-181252/"
        self.model_path = "/cache/user-job-dir/MassFac/models/"
        self.output_graph = "/cache/user-job-dir/MassFac/models/massface1.pb"

    def ckpt2pb2(self):
        #network = importlib.import_module('models.covpoolnet2')
        #network = importlib.import_module('.models.model-20211130-163207')
        inputX = tf.placeholder(tf.float32, shape=[1, 112, 112, 3],
                                name='inputImage')

        prelogits, net_points = mobilenet.inference(inputX, bottleneck_layer_size=1024,
                                                        phase_train=False, weight_decay=1e-4,reuse=False)

        prediction = tf.identity(prelogits,name='output')

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=["output"])

            with tf.gfile.GFile(self.output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())


    def convert_pb(self):
        # read graph definition
        f = tf.io.gfile.GFile(self.output_graph, "rb")
        gd = graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # fix nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, self.model_path , 'massface.pb', as_text=False)


if __name__ == "__main__":
    temp = getinfo()
    temp.ckpt2pb2()
    temp.convert_pb()
    print("done")
    mox.file.copy_parallel(src_url="/cache/user-job-dir/MassFac/models/massface1.pb",
                           dst_url="obs://qyy-unet/massface2.pb")
    print("done")
    # mox.file.copy_parallel(src_url="models/facenet_ms_mp/", dst_url="obs://2021-buckets-test2/MassFac/facenet_ms_mp/")
