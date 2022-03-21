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

# from create_tf_record import *
from tensorflow.python.framework import graph_util
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


from tensorflow.python.platform import gfile
model_path = "models/frozen_model.pb"

######################
model_dir = "models/"

ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path
print_tensors_in_checkpoint_file(ckpt_path, all_tensors=True, all_tensor_names=True, tensor_name='')

######################

###########查看pbnode##########
# # read graph definition
# f = gfile.FastGFile(model_path, "rb")
# gd = graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())
#
# # fix nodes
# for node in graph_def.node:
#     if node.op == 'RefSwitch':
#         node.op = 'Switch'
#         for index in range(len(node.input)):
#             if 'moving_' in node.input[index]:
#                 node.input[index] = node.input[index] + '/read'
#     elif node.op == 'AssignSub':
#         node.op = 'Sub'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#     elif node.op == 'Assign':
#         node.op = 'Identity'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#         if 'validate_shape' in node.attr: del node.attr['validate_shape']
#         if len(node.input) == 2:
#             # input0: ref: Should be from a Variable node. May be uninitialized.
#             # input1: value: The value to be assigned to the variable.
#             node.input[0] = node.input[1]
#             del node.input[1]
#     elif node.op == 'AssignAdd':
#         node.op = 'Add'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#
# # import graph into session
# tf.import_graph_def(graph_def, name='')
# for i, n in enumerate(graph_def.node):
#     print("=====node====")
#     print("Name of the node - %s" % n.name)
# #######################


# # tensorflow查看ckpt各节点名称
# checkpoint_path=os.path.join('models/gan.ckpt')
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print ('tensor_name: ', key)
# def freeze_graph_test(pb_path, image_path):
#     '''
#     :param pb_path:pb文件的路径
#     :param image_path:测试图片的路径
#     :return:
#     '''
#     with tf.Graph().as_default():
#         output_graph_def = tf.GraphDef()
#         with open(pb_path, "rb") as f:
#             output_graph_def.ParseFromString(f.read())
#             tf.import_graph_def(output_graph_def, name="")
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#
#             # 定义输入的张量名称,对应网络结构的输入张量
#             # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
#             input_image_tensor = sess.graph.get_tensor_by_name("input:0")
#             input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
#             input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
#
#             # 定义输出的张量名称
#             output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
#
#             # 读取测试图片
#             im = read_image(image_path, resize_height, resize_width, normalization=True)
#             im = im[np.newaxis, :]
#             # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
#             # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
#             out = sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
#                                                           input_keep_prob_tensor: 1.0,
#                                                           input_is_training_tensor: False})
#             print("out:{}".format(out))
#             score = tf.nn.softmax(out, name='pre')
#             class_id = tf.argmax(score, 1)
#             print
#             "pre class_id:{}".format(sess.run(class_id))


# def freeze_graph(input_checkpoint, output_graph):
#     '''
#     :param input_checkpoint:
#     :param output_graph: PB模型保存路径
#     :return:
#     '''
#     # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
#     # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
#
#     # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
#     output_node_names = "DISCRIMINATOR_CP/h1_lin/b"
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     graph = tf.get_default_graph()  # 获得默认的图
#     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
#
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
#         output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
#             sess=sess,
#             input_graph_def=sess.graph_def,  # 等于:sess.graph_def
#             output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
#
#         with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
#             f.write(output_graph_def.SerializeToString())  # 序列化输出
#         print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
#
#         # for op in sess.graph.get_operations():
#         #     print(op.name, op.values())

checkpoint_path=os.path.join('models/gan.ckpt')
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    a=reader.get_tensor(key)
    print( 'tensor_name: ',key)
    print("a.shape:%s"%[a.shape])

def freeze_graph2(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = 'GENERATOR/output'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        tensor_name_list = [tensor.name for tensor in
                            tf.get_default_graph().as_graph_def().node]
        t_name_list = []
        for names in tensor_name_list:
            if names == output_node_names:
                t_name_list.append(names)
                break
            else:
                t_name_list.append(names)
        print("tensor_name_list:", t_name_list)
        # t_name_list.remove('real_points_ph')
        # t_name_list.remove('fake_points_ph')

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=t_name_list)  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
        print("###########")
        # for op in graph.get_operations():
        #     print(op.name, op.values())


def freeze_graph_test(pb_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    f = gfile.FastGFile(pb_path, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

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

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            # tf.import_graph_def(output_graph_def, name="")
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("noise_ph:0")
            # input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            # input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("GENERATOR/output:0")
            # data = np.loadtxt("models/noise_data.txt")  # 将文件中数据加载到data数组里
            data = np.load(r"./noise.npy")
            data.tofile("noise.bin")
            print(data)

            # data_image = np.loadtxt("models/image_data.txt")
            # np.save("image_data.npy", data_image)
            # # 读取测试图片
            # im = read_image(image_path, resize_height, resize_width, normalization=True)
            # im = im[np.newaxis, :]
            # # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})

            out = sess.run(output_tensor_name, feed_dict={input_image_tensor:data})
            print("=======3333======")
            print("out:{}".format(out))
            print(out.shape)
            np.save("out.npy",out)
            print("pb output length:", len(out))
            # score = tf.nn.softmax(out, name='pre')
            # class_id = tf.argmax(score, 1)
            # print
            # "pre class_id:{}".format(sess.run(class_id))


if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = 'models/gan.ckpt'
    # 输出pb模型的路径
    out_pb_path = "models/frozen_model.pb"
    # 调用freeze_graph将ckpt转为pb
    # freeze_graph2(input_checkpoint, out_pb_path)
    freeze_graph_test(out_pb_path)


    # 测试pb模型
    # image_path = 'test_image/animal.jpg'
    # freeze_graph_test(pb_path=out_pb_path, image_path=image_path)