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

import argparse
from PIL import Image
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import tensorflow as tf
import numpy as np
import time
import os
def parse_args():
    '''

    :return:
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='stnet.pb',
                        help="""pb path""")
    parser.add_argument('--input_tensor_name', default='Placeholder:0',
                        help="""input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='output_logit:0',
                        help="""output_tensor_name""")
    parser.add_argument('--data_url', default="./dataset/mnist_sequence1_sample_5distortions5x5.npz",
                        help="""the label data path""")
    parser.add_argument('--inference_url', default="./out/",
                        help="""the bin file path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def read_file(image_name, path):
    '''
    read image label from label_file
    :param image_name:
    :param path:
    :return:
    '''
    with open(path, 'r') as cs:
        rs_list = cs.readlines()
        for name in rs_list:
            if image_name in str(name):
                num = str(name).split(" ")[1]
                break
    return int(num) + 1

def normalize(inputs):
    '''

    :param inputs:
    :return:
    '''
    mean = [121.0, 115.0, 100.0]
    std =  [70.0, 68.0, 71.0]
    mean = tf.expand_dims(tf.expand_dims(mean, 0), 0)
    std = tf.expand_dims(tf.expand_dims(std, 0), 0)
    inputs = inputs - mean
    inputs = inputs * (1.0 / std)
    return inputs


class Actor(object):
    # set batchsize:
    args = parse_args()
    # batch_size = int(args.batchsize)

    def __init__(self):


        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/app/smith_model/spatial-transformer-tensorflow_npu_20211113193745/fusion_switch.cfg")

        custom_op.parameter_map["use_off_line"].b = True

        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        custom_op.parameter_map["graph_run_mode"].i = 0

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load frozen graph
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch):
        """
        do infer
        :param image_data:
        :return:
        """
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                y = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch})
        return np.argmax(y)

def main():
    args = parse_args()
    tf.reset_default_graph()

    print("########NOW Start Preprocess!!!#########")
    mnist_cluttered = np.load(args.data_url)
    y_train = mnist_cluttered['y_test']
    tot =len(os.listdir(args.inference_url))
    acc =0
    print("########NOW Start inference!!!#########")
    cost_time=0
    
    for i in range(tot):
        batch = np.fromfile(args.inference_url+"%d.bin" %i, dtype=np.float32)
        batch =batch.reshape(1 ,1600)
        actor = Actor()
        st =time.time()
        res =actor.do_infer(batch)
        ed =time.time()
        cost_time+=ed-st
        if res==y_train[i]:
            acc =acc +1
    print('======acc : {}----total : {}'.format(acc, tot))
    print('Final Online Inference Accuracy accuracy : ', round(acc / tot, 4))
    print("cost time:", cost_time)
    print("average infer time:{0:0.3f} ms/img,FPS is {1:0.3f}".format(cost_time * 1000 / tot, 1 / (cost_time / tot)))
if __name__ == '__main__':
    main()
