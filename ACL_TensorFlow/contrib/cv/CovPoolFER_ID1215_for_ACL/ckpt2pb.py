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
import tensorflow.contrib.slim as slim
import argparse,sys
class getinfo:
    def __init__(self,ckpt_path,output_path):
        #self.model_path = "/home/lf/models/20211231-181827/"
        self.model_path    = ckpt_path
        self.output_graph  = output_path
    def ckpt2pb(self):
        network = importlib.import_module("models.covpoolnet3")
        inputX = tf.placeholder(tf.float32, shape=[128, 100, 100, 3],
                                name='inputImage')

        prelogits = network.inference(inputX, keep_probability=1.0, phase_train=False,
                                      bottleneck_layer_size=128,
                                      weight_decay=0.05)
        logits = slim.fully_connected(prelogits, 7 ,activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.05),
                                      scope='Logits', reuse=False)
        prediction = tf.argmax(input=logits, axis=-1, output_type=tf.dtypes.int32, name="output")
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

        print("done")
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

        # import graph into session
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, self.output_graph , 'CovPoolFER.pb', as_text=False)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        help='file where to load ckpt.',default= "/home/lf/models/20220115-035551/")
    parser.add_argument('--output_path', type=str,
                        help='file where to output pb.',default= "/home/lf/models/")
    return parser.parse_args(argv)
def main(args):
    temp = getinfo(args.ckpt_path,args.output_path)
    temp.ckpt2pb()
    temp.convert_pb()
if __name__ == "__main__" :
    main(parse_arguments(sys.argv[1:]))