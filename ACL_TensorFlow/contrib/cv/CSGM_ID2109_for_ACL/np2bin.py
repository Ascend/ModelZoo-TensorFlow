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


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
# mnist_data = '/home/dataset/mnist'
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

def freeze_graph_test(pb_path, image):
    '''
    :param pb_path:pb文件的路径
    :param image:测试图片
    :return:
    '''
    #===============验证pb模型=================
    f = gfile.FastGFile(pb_path, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            # tf.import_graph_def(output_graph_def, name="")
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("x_ph:0")
            input_tensor = sess.graph.get_tensor_by_name("x_ph_1:0")
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("gen/output:0")
            im = image
            out1 = sess.run(output_tensor_name, feed_dict={input_image_tensor: im,input_tensor:z_val})
            print("out:{}".format(out1))
            print(out1[0:10])
            print("pb output length:",len(out1))
            predict = np.array(out1)
            reconstruct_error = 0
            reconstruct_error += np.sum(np.square(np.linalg.norm(predict - test_data)))
            reconstruct_error = reconstruct_error / (784 * 100)
            print("re:",reconstruct_error)
            print(np.sum(test_data[0]-predict[0])/784)
            # print(predict[0])

            # print("out:{}".format(out2))


test_data.tofile("test_img.bin") # 处理后的图片保存为bin文件
z_val.tofile("test_z.bin")
pb_path = './frozen_model.pb'
image = test_data
freeze_graph_test(pb_path, image)