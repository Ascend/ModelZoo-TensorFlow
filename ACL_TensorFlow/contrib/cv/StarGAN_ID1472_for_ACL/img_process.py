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
测试数据预处理与推理数据后处理
"""

import os
import shutil
import numpy as np
from tqdm import tqdm
import data_loader
import argparse


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


def img_preprocess(image_root, metadata_path, save_dir, c_dim=5, batch_size=1, selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    # 创建文件路径
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(os.path.join(save_dir, "x_real"))
    os.makedirs(os.path.join(save_dir, "c_trg"))

    # 获取测试图片
    dataset_test = data_loader.CelebADataset(image_root=image_root, metadata_path=metadata_path,
                                             is_training=False, batch_size=batch_size,
                                             image_h=128, image_w=128,
                                             image_c=3)
    data_generate_test = dataset_test.batch_generator_numpy()

    # 保存预处理生成的bin文件
    for i in tqdm(range(2000)):
        data_gen_test = next(data_generate_test)
        x_real = data_gen_test["images"]
        c_org = data_gen_test["attribute"]
        c_trg_list = create_labels(c_org, c_dim, 'CelebA', selected_attrs.split(" "))

        j = 0
        for c_trg in c_trg_list:
            # 生成图像文件
            x_real.tofile(os.path.join(save_dir, "x_real", "{}_{}.bin".format(i, j)))
            # 生成属性文件
            c_trg = c_trg.astype(np.float32)  # bool转float32
            c_trg.tofile(os.path.join(save_dir, "c_trg", "{}_{}.bin".format(i, j)))
            j += 1

    print("img_preprocess finished ...")


def img_postprocess(input_images, input_result_om, result_dir, c_dim=5):
    '''
    :param input_images:原始图像数据的路径
    :param input_result_om:om推理生成数据的路径
    :param result_dir:后处理保存图像的路径
    :return:
    '''
    # 创建文件路径
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    os.makedirs(os.path.join(result_dir, "src_images"))
    os.makedirs(os.path.join(result_dir, "generates"))

    for i in tqdm(range(2000)):
        x_real_concat = []
        fake_images_concat = []
        for j in range(c_dim):
            # cpnvert src images
            x_real = np.fromfile(os.path.join(input_images, "{}_{}.bin".format(i, j)), dtype='float32')
            x_real.shape = 1, 128, 128, 3
            x_real_concat.append(x_real)

            # convert generates images
            fake_images = np.fromfile(os.path.join(input_result_om, "{}_{}_output_0.bin".format(i, j)), dtype='float32')
            fake_images.shape = 1, 128, 128, 3
            fake_images_concat.append(fake_images)

        # save src images
        x_concat = np.concatenate(x_real_concat, axis=2)
        result_path = os.path.join(result_dir, 'src_images/{}-images.jpg'.format(i))
        data_loader.save_images(x_concat, result_path, 1)

        # save generates images
        x_concat = np.concatenate(fake_images_concat, axis=2)
        result_path = os.path.join(result_dir, 'generates/{}-images.jpg'.format(i))
        data_loader.save_images(x_concat, result_path, 1)

    print("img_postprocess finished ...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="img_preprocess", help="model phase: img_preprocess/img_postprocess.")
    parser.add_argument("--image_root", default="./datasets/celeba/images", help="the root path of images.")
    parser.add_argument("--metadata_path", default="./datasets/celeba/list_attr_celeba.txt", help="the path of metadata.")
    parser.add_argument("--save_dir", default='./test_data', help="the path of output bin files.")
    parser.add_argument("--selected_attrs", default="Black_Hair Blond_Hair Brown_Hair Male Young", help="selected attributes for the CelebA dataset.")
    parser.add_argument("--c_dim", default=5, help="the dimension of condition.")
    parser.add_argument("--batch_size", default=1, help="batch size of data.")
    parser.add_argument("--input_images_bin", default="./test_data/x_real", help="the root path of images.")
    parser.add_argument("--input_results_om_bin", default="./test_data/outputs_om", help="the root path of images.")
    parser.add_argument("--result_dir", default="./results_om", help="the root path of images.")
    args = parser.parse_args()

    # 参数初始化
    phase = args.phase
    image_root = args.image_root
    metadata_path = args.metadata_path
    save_dir = args.save_dir
    selected_attrs = args.selected_attrs
    c_dim = args.c_dim
    batch_size = args.batch_size

    input_images_bin = args.input_images_bin
    input_results_om_bin = args.input_results_om_bin
    result_dir = args.result_dir


    # 数据预处理
    if (phase == "img_preprocess"):
        img_preprocess(image_root, metadata_path, save_dir, c_dim=c_dim, batch_size=batch_size, selected_attrs=selected_attrs)


    # 数据后处理
    if (phase == "img_postprocess"):
        img_postprocess(input_images_bin, input_results_om_bin, result_dir)

