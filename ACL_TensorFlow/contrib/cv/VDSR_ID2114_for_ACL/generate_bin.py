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
import numpy as np
import os, argparse, glob, re
import moxing as mox
import scipy.io
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/vdsr-jcr/data/test/")#obs该目录下放4个测试集文件夹（B100、Set14、Set5、Urban100）
parser.add_argument("--train_url", type=str, default="/vdsr-jcr/")
args = parser.parse_args()
data_dir = "/cache/dataset"
os.makedirs(data_dir)
mox.file.copy_parallel(args.data_url, data_dir)
a = os.path.join(args.data_url, "B100")
b = os.path.join(args.data_url, "Set14")
c = os.path.join(args.data_url, "Set5")
d = os.path.join(args.data_url, "Urban100")

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
def g_bin(inpath,outpath):
    folder_list = glob.glob(inpath)
    for folder_path in folder_list:
        img_list = get_img_list(folder_path)
        for i in range(len(img_list)):
            input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            input_tensor_1 = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            input_tensor_5 = tf.zeros_like(input_tensor_1, dtype=tf.float32)
            input_tensor_4 = tf.add(input_tensor_1, input_tensor_5)
            input_tensor_2 = np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))
            input_tensor = tf.Session().run(input_tensor_4, feed_dict={input_tensor_1: input_tensor_2})

            se = ''.join('%s' %j for j in img_list[i])
            ss1 = se.split("/")[7].split(".")[0]
            ss2 = se.split("/")[14].split(".")[0]
            input_tensor.tofile(args.data_url + "here/" + outpath + "/" + ss2 + ".bin")
            input_tensor.tofile(args.data_url + "here/" + outpath + "/" + ss1 + ".bin")

g_bin(a, "B100")
g_bin(b, "Set14")
g_bin(c, "Set5")
g_bin(d, "Urban100")
mox.file.copy_parallel(args.data_url, args.train_url)#here文件夹下得到转好的4个测试集