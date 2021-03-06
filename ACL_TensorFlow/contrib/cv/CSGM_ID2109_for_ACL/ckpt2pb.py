#!/usr/bin/env python 
# -*- coding:utf-8 -*-

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


import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from tensorflow_core.python.framework import graph_util
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_dir = 'model/datasets'
mnist_data = data_dir
mnist = input_data.read_data_sets(mnist_data, one_hot=True)  # ./data/mnist
test_images = mnist.test.images
print("test_data", test_images.shape)
print(type(test_images))
test_index = np.random.randint(0,1000,size=(100,))
test_data = test_images[test_index]
print(test_data.dtype)
z_val = np.random.randn(100, 20)
z_val = z_val.astype(np.float32)
print(z_val.shape)
print(z_val.dtype)


def print_tensors(pb_file):
    print('Model File: {}\n'.format(pb_file))
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name + '\t' + str(op.values()))

def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB??????????????????
    :return:
    '''

    # ???????????????????????????,???????????????????????????????????????????????????
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # ????????????????????????
        #??????ckpt???????????????????????????????????????
        graph_def = tf.get_default_graph()

        x = graph_def.get_tensor_by_name("x_ph:0")
        x1 = graph_def.get_tensor_by_name("x_ph_1:0")
        logits = graph_def.get_tensor_by_name("gen/output:0:0")
        pre = sess.run(logits,{x:test_data,x1:z_val})
        print("pre:",pre)
        predict = np.array(pre)

        reconstruct_error = np.sum(np.square(np.linalg.norm(predict - test_data)))
        reconstruct_error = reconstruct_error / (784 * 100)
        print("re:", reconstruct_error)



        #??????????????????
        tensor_name_list = [tensor.name for tensor in
                            tf.get_default_graph().as_graph_def().node]

        t_name_list = ['x_ph','x_ph_1']
        for names in tensor_name_list:
            if names == "gen/output:0":
                t_name_list.append(names)
                break
            else:
                t_name_list.append(names)
        print("tensor_name_list:", tensor_name_list)

        graph_def = sess.graph_def
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                
        output_graph_def = graph_util.convert_variables_to_constants(  # ????????????????????????????????????
            sess=sess,
            input_graph_def=graph_def,  # ??????:sess.graph_def
            output_node_names=t_name_list)  # ?????????????????????????????????????????????
        with tf.gfile.GFile(output_graph, "wb") as f:  # ????????????
            f.write(output_graph_def.SerializeToString())  # ???????????????
        print("%d ops in the final graph." % len(output_graph_def.node))  # ????????????????????????????????????





# ??????ckpt????????????
input_checkpoint='D:\PycharmProjects\sklearnDemo\model\csgm.ckpt'

# ??????pb???????????????
out_pb_path="frozen_model.pb"
# ??????freeze_graph???ckpt??????pb
freeze_graph(input_checkpoint,out_pb_path)
print_tensors("./frozen_model.pb")


#atc??????
# atc --model=/usr/model_test/frozen_model.pb --framework=3 --output=/usr/model_test/frozen_model
# --soc_version=Ascend310 --out_nodes="gen_1/Sigmoid:0" --input_shape "x_ph:100,784;x_ph_1:100,20"

#mame??????

# git clone https://gitee.com/ascend/tools.git
# mv tools /home/HwHiAiUser/AscendProjects/
# export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
# export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub

# cd /home/HwHiAiUser/AscendProjects/tools/msame/
# ./build.sh g++ /home/HwHiAiUser/AscendProjects/tools/msame/out

#om??????
# ./msame --model "/usr/model_test/frozen_model.om" --input "/usr/model_test/test_img.bin,/usr/model_test/test_z.bin" --output "/usr/model_test/output/" --outfmt TXT  --loop 1





