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
# ============================================================================
import tensorflow as tf
import json
import numpy as np
from load_data_multimodal import load_graph, load_fitb_data1, load_auc_data1
from datetime import *

def freeze_graph_test(pb_path, data_dir, batch_size, image_feature_path, text_feature_path):
    '''
    :param pb_path:pb文件的路径
    :param data_dir:测试outfit json的路径
    :param batch_size:设置为16
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #print(sess.run(sess.graph.get_tensor_by_name("gnn_image/w/in_image_0:0")))

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            input_text_tensor = sess.graph.get_tensor_by_name("Placeholder_2:0")
            input_graph = sess.graph.get_tensor_by_name("Placeholder_4:0")

            # 定义输出的张量名称 is ok
            output_tensor_name = sess.graph.get_tensor_by_name("s_pos_output:0")

            # 读取数据
            # per_outfit = 8
            # G = load_graph()
            ftest = open(data_dir, 'r')  # 测试数据
            test_outfit_list = json.load(ftest)
            test_size = len(test_outfit_list)
            ############test############
            batches_fitb = int((test_size * 4) / batch_size)
            right = 0.
            for i in range(batches_fitb):
                test_fitb = load_fitb_data1(i, batch_size, test_outfit_list, image_feature_path, text_feature_path)
                answer = sess.run([output_tensor_name], feed_dict={
                    input_image_tensor: test_fitb[0], input_text_tensor: test_fitb[1], input_graph: test_fitb[2]})
                #print(test_fitb[2])

                answer = np.asarray(answer[0])

                for j in range(int(batch_size / 4)):
                    a = []
                    for k in range(j * 4, (j + 1) * 4):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.

                if i % 10 == 0:
                    print("acc batch: " + str(i) + " is over!")
                    #print(sess.run(sess.graph.get_tensor_by_name("gnn_image/w/in_image_0:0")))


            print(answer)
            accurancy = float(right / test_size)

            ##### AUC #####
            batches = int((test_size * 2) / batch_size)
            right = 0.
            for i in range(batches):
                test_auc = load_auc_data1(i, batch_size, test_outfit_list, image_feature_path, text_feature_path)
                answer = sess.run([output_tensor_name], feed_dict={input_image_tensor: test_auc[0],
                                                      input_text_tensor: test_auc[1],
                                                      input_graph: test_auc[2]})
                answer = np.asarray(answer[0])

                for j in range(int(batch_size / 2)):
                    a = []
                    for k in range(j * 2, (j + 1) * 2):
                        a.append(answer[k][0])
                    if np.argmax(a) == 0:
                        right += 1.
                    #print(test_auc[2])

                if i % 10 == 0:
                    print("auc batch: " + str(i) + " is over!")

            print(answer)
            auc = float(right / test_size)

            print('now():' + str(datetime.now()))
            print("Test:",  "accuracy:", "{:.9f}".format(accurancy), "auc:", "{:.9f}".format(auc))

            # for i in range(test_batches):
            #     image, text = load_data(i, batch_size, test_list, image_feature_path, text_feature_path)
            #     # dis_pos_ = sess.run([s_pos_mean],feed_dict={image_pos: image_pos_,text_pos: text_pos_,graph_pos: G})
            #     # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            #     # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            #     out = sess.run(output_tensor_name, feed_dict={
            #         input_image_tensor: image, input_text_tensor: text, input_graph: G})
            #     out = np.asarray(out[0])
            #     print("out:{}".format(out))




if __name__ == '__main__':

    # 输出pb模型的路径
    out_pb_path = "./pb_model/frozen_model_last.pb"
    # 测试pb模型
    data_dir = "./data/test_no_dup_new_100.json"
    batch_size=16
    image_feature_path="F:/NGNN/data/polyvore_image_vectors/"
    text_feature_path="F:/NGNN/data/polyvore_text_onehot_vectors/"
    freeze_graph_test(out_pb_path, data_dir, batch_size, image_feature_path, text_feature_path)
