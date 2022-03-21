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
from tensorflow.python.platform import gfile
import numpy as np
import os, argparse, glob, re
import moxing as mox
import scipy.io
from PSNR import psnr

def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat", 4])
    return train_list
def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:
        #print (pair[1])
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if "img_2" in mat_dict:
            input_img = mat_dict["img_2"]
        elif "img_3" in mat_dict:
            input_img = mat_dict["img_3"]
        elif "img_4" in mat_dict:
            input_img = mat_dict["img_4"]
        else:
            continue
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list

def freeze_graph_test(pb_path, folder):
    '''
    :param pb_path:pb文件的路径
    :param image:测试图片
    :return:
    '''
    # read graph definition
    f = gfile.FastGFile(pb_path, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name("fifo_queue_DequeueMany:0")#指定pb模型的输入节点
            output_tensor = sess.graph.get_tensor_by_name("shared_model/Add:0")#指定pb模型的输出节点
            folder_list = glob.glob(folder)
            for folder_path in folder_list:
                img_list = get_img_list(folder_path)
                for i in range(len(img_list)):
                    input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
                    input_y = input_list[0]

                    input_tensor_1 = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                    input_tensor_5 = tf.zeros_like(input_tensor_1, dtype=tf.float32)
                    input_tensor_4 = tf.add(input_tensor_1,input_tensor_5)
                    input_tensor_2 = np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))
                    input_tensor_3 = sess.run(input_tensor_4, feed_dict={input_tensor_1:input_tensor_2})
                    out1 = sess.run(output_tensor, feed_dict={input_tensor:input_tensor_3})
                    # 精度测试
                    gt_y = gt_list[0]
                    img_vdsr_y = np.resize(out1, (input_y.shape[0], input_y.shape[1]))
                    psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
                    print("VDSR %f" % ( psnr_vdsr))

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/vdsr-jcr/data/pbtest/")#obs该目录下放测试图片文件夹（这里用set5，下面有20张mat格式图片）、测试的pb模型
parser.add_argument("--train_url", type=str, default="/vdsr-jcr/")
args = parser.parse_args()
data_dir = "/cache/dataset"
os.makedirs(data_dir)
mox.file.copy_parallel(args.data_url, data_dir)

pb_path = os.path.join(args.data_url, "VDSR_shared_model_Add_queue.pb")#需调整
image = os.path.join(args.data_url, "Set5")#需调整
freeze_graph_test(pb_path, image)