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
import tensorflow as tf
import importlib
import os
import numpy as np
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 测试pb文件使用是否正常
def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("sr_input:0")
            input_image_scale = sess.graph.get_tensor_by_name("sr_input_scale:0")

            # output_list = [sess.graph.get_tensor_by_name("generator/add:0")]
            # for i in range(1, 16):
            #     op = sess.graph.get_tensor_by_name("generator/add_{}:0".format(str(i)))
            #     output_list.append(op)

            # output = tf.concat(output_list, axis=0, name="output")
            output = sess.graph.get_tensor_by_name("output:0")

            # 读取测试图片,此处使用假数据
            im = np.random.randn(10, 60, 60, 3)

            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字

            out = sess.run(output, feed_dict={input_image_tensor: im,
                                              input_image_scale: 4})
            out1 = tf.split(out, 16, axis=0)
            print("out:{}".format(out.shape))
            print(im.shape)

            for i in out1:
                print(i.shape)


def freeze_graph(input_checkpoint, output_graph):
    """
    :param input_checkpoint: ckpt模型路径
    :param output_graph: PB模型保存路径
    :return:
    """

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "output"
    # for i in range(1, 16):
    #     output_node_names += ",generator/add_{}".format(i)

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_list = [sess.graph.get_tensor_by_name("generator/add:0")]
        print("outputlist:", output_list)
        for i in range(1, 16):
            op = sess.graph.get_tensor_by_name("generator/add_{}:0".format(str(i)))
            output_list.append(op)

        finaloutput = tf.concat(output_list, axis=0, name="output")
        print("finaloutput:", finaloutput)
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = './temp/results/model.ckpt-1000000'
    # 输出pb模型的路径
    out_pb_path = "./frozen_model.pb"
    # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint, out_pb_path)

    # # 测试pb模型
    image_path = ''
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)
