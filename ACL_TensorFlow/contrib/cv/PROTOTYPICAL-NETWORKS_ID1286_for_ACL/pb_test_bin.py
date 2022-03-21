# coding: gbk
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

"""
    Here, we show the method to test the model using *.pb file
    with *.bin file as the test data.
"""
from tensorflow.python.platform import gfile
import tensorflow as tf
from PIL import Image
import numpy as np

n_way = 60
n_shot = 5
n_query = 5
n_examples = 20
im_width = 28
im_height = 28
channels = 1
h_dim = 64
z_dim = 64

n_test_episodes = 1000
n_test_way = 20
n_test_shot = 5
n_test_query = 15

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open('./checkpoint/log/proto_model.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Defines the name of the input nodes.
        input_x = sess.graph.get_tensor_by_name("inputx:0")
        input_q = sess.graph.get_tensor_by_name("inputq:0")
        input_y = sess.graph.get_tensor_by_name("inputy:0")

        # Defines the name of the output node.
        output_tensor_name = sess.graph.get_tensor_by_name("output:0")

        # testing...
        avg_acc = 0.
        for epi in range(n_test_episodes):
            inputx_len = 20*5*28*28*1
            inputq_len = 20*15*28*28*1
            inputy_len = 20*15
            inputx_name = "./input/inputx/input"+ str(epi) + ".bin"
            inputq_name = "./input/inputq/input"+ str(epi) + ".bin"
            inputy_name = "./input/inputy/input"+ str(epi) + ".bin"
            
            support_test = np.fromfile(inputx_name, dtype=np.float32).reshape(20, 5, 28, 28, 1)
            query_test = np.fromfile(inputq_name, dtype=np.float32).reshape(20, 15, 28, 28, 1)
            labels_test = np.fromfile(inputy_name, dtype=np.int64).reshape(20, 15)
                        
            # print(support_test.shape)
            # print(query_test.shape)
            # print(labels_test.shape)
            ac_test = sess.run(output_tensor_name, feed_dict={input_x: support_test, input_q: query_test, input_y: labels_test})
            avg_acc += ac_test
            if (epi + 1) % 50 == 0:
                 print('[test episode {}/{}] => acc: {:.5f}'.format(epi + 1, n_test_episodes, ac_test))
        avg_acc /= n_test_episodes
        print('Average Test Accuracy: {:.5f}'.format(avg_acc))
