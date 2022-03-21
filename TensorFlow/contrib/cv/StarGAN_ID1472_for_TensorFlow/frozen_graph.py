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
"""
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

import os
import shutil
from tqdm import tqdm
from data import data_loader
import argparse


def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = np.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.copy()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(np.ones(np.array(c_org).shape(0))*i, c_dim)

        c_trg_list.append(c_trg)

    return c_trg_list

def freeze_graph(input_checkpoint, output_graph):
    '''

    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "G/Tanh"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    # show_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, input_checkpoint)  #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        show_all_variables()

        with tf.gfile.GFile(output_graph, "wb") as f:  #保存模型
            f.write(output_graph_def.SerializeToString())  #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  #得到当前图有几个操作节点

        # for op in sess.graph.get_operations():
        #     print(op.name, op.values())

    print("freeze_graph finished ...")


def freeze_graph_test(pb_path, image_root, metadata_path, result_dir, c_dim=5, batch_size=1, selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young", test_concatenate=True):
    with tf.Graph().as_default():
        # npu config
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # Placeholder:0作为输入图像, Placeholder_4:0作为输入标签
            input_images_tensor = sess.graph.get_tensor_by_name("x_real:0")
            input_labels_tensor = sess.graph.get_tensor_by_name("c_trg:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("G/Tanh:0")

            # 创建文件路径
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)

            if test_concatenate:
                os.makedirs(os.path.join(result_dir, "src_images"))
                os.makedirs(os.path.join(result_dir, "generates"))
                os.makedirs(os.path.join(result_dir, "x_real"))
                os.makedirs(os.path.join(result_dir, "c_trg"))
                os.makedirs(os.path.join(result_dir, "outputs_pb"))
            else:
                os.makedirs(os.path.join(result_dir, "src"))
                for c_ in range(1, c_dim + 1):
                    os.makedirs(os.path.join(result_dir, "generate{}".format(c_)))

            # 获取测试图片
            dataset_test = data_loader.CelebADataset(image_root=image_root, metadata_path=metadata_path,
                                                     is_training=False, batch_size=batch_size,
                                                     image_h=128, image_w=128,
                                                     image_c=3)
            data_generate_test = dataset_test.batch_generator_numpy()

            for i in tqdm(range(20)):
                data_gen_test = next(data_generate_test)
                x_real = data_gen_test["images"]
                c_org = data_gen_test["attribute"]
                c_trg_list = create_labels(c_org, c_dim, 'CelebA', selected_attrs.split(" "))

                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    c_feed_dict = {
                        # numpy ndarray
                        input_images_tensor: x_real,
                        input_labels_tensor: c_trg,
                    }
                    x_real.tofile(os.path.join(result_dir, "x_real", "{0:05d}.bin".format(i)))
                    c_trg.tofile(os.path.join(result_dir, "c_trg", "{0:05d}.bin".format(i)))
                    fake_images = sess.run(output_tensor_name, feed_dict=c_feed_dict)
                    fake_images.tofile(os.path.join(result_dir, "outputs_pb", "{0:05d}.bin".format(i)))
                    x_fake_list.append(fake_images)

                if test_concatenate:

                    # save src images
                    src_images = [x_real for _ in range(c_dim)]

                    x_concat = np.concatenate(src_images, axis=2)
                    result_path = os.path.join(result_dir, 'src_images/{}-images.jpg'.format(i + 1))
                    data_loader.save_images(x_concat, result_path, batch_size)

                    # save generates images
                    x_concat = np.concatenate(x_fake_list[1:], axis=2)
                    result_path = os.path.join(result_dir, 'generates/{}-images.jpg'.format(i + 1))
                    data_loader.save_images(x_concat, result_path, batch_size)
                else:
                    # save image only
                    x_concat = x_fake_list[0]
                    result_path = os.path.join(result_dir, 'src/{}-images.jpg'.format(i + 1))
                    data_loader.save_images(x_concat, result_path, batch_size)
                    # Black_Hair Blond_Hair Brown_Hair Male Young
                    for c_ in range(1, c_dim + 1):
                        x_concat = x_fake_list[c_]
                        result_path = os.path.join(result_dir, 'generate{}/{}-images.jpg'.format(c_, i + 1))
                        data_loader.save_images(x_concat, result_path, batch_size)

    print("freeze_graph_test finished ...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="frozen_graph", help="model phase: frozen_graph/test_pb.")
    parser.add_argument("--input_checkpoint", default="./ckpt/model-200000", help="the path of checkpoint file.")
    parser.add_argument("--out_pb_path", default="./pb/stargan_model.pb", help="the path of pb file.")
    parser.add_argument("--image_root", default="./datasets/celeba/images", help="the root path of images.")
    parser.add_argument("--metadata_path", default="./datasets/celeba/list_attr_celeba.txt", help="the path of metadata.")
    parser.add_argument("--result_dir", default="./results_pb", help="the path for results of pb.")
    parser.add_argument("--selected_attrs", default="Black_Hair Blond_Hair Brown_Hair Male Young", help="selected attributes for the CelebA dataset.")
    parser.add_argument("--c_dim", default=5, help="the dimension of condition.")
    parser.add_argument("--batch_size", default=1, help="batch size of data.")
    args = parser.parse_args()

    phase = args.phase
    input_checkpoint = args.input_checkpoint
    out_pb_path = args.out_pb_path
    image_root = args.image_root
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    selected_attrs = args.selected_attrs
    c_dim = args.c_dim
    batch_size = args.batch_size

    # 生成pb模型
    if(phase == "frozen_graph"):
        freeze_graph(input_checkpoint, out_pb_path)

    # 测试pb模型
    if (phase == "test_pb"):
        freeze_graph_test(pb_path=out_pb_path, image_root=image_root, metadata_path=metadata_path, result_dir=result_dir, c_dim=c_dim, batch_size=batch_size, selected_attrs=selected_attrs)

