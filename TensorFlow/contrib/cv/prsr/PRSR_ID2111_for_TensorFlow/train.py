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
import sys
import os
import subprocess
#import precision_tool.config as CONFIG

# subprocess.call(["export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/$LD_PRELOAD"])

#-----------------------打印路径-------------------
print('当前路径为: {}'.format(os.path.abspath(__file__)))
# print('数据路径为: {}'.format(os.listdir('/home/ma-user/modelarts/inputs/data_url_0')))
#------------------------------------------------

sys.path.insert(0, './')

flags = tf.app.flags

#-----------------------上载OBS文件至ModelArts------------
import argparse
#import moxing as mox
# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--train_url", type=str, default="./output")
parser.add_argument("--data_url", type=str, default="/celeba-mytest/celebA_test")
config, unparsed = parser.parse_known_args()
# 在ModelArts容器创建数据存放目录
# os.makedirs("/cache/overflow_data")
# data_dir = "/cache/dataset"
# os.makedirs(data_dir)
# OBS数据拷贝到ModelArts容器内
# mox.file.copy_parallel(config.data_url, data_dir)
# #------------------------------------------------------------

# print('数据路径2为: {}'.format(os.listdir('/cache/dataset')))

# #---------------改写txt文件-------------------
# # options.dataset = '/root/data/celebA'
# # options.outfile = '/root/data/train.txt'
# # dataset = '/home/ma-user/modelarts/inputs/data_url_0'
# dataset = data_dir
# outfile = '/home/ma-user/modelarts/user-job-dir/code/train.txt'

# f = open(outfile, 'w')
# dataset_basepath = dataset
# for p1 in os.listdir(dataset_basepath):
#   image = os.path.abspath(dataset_basepath + '/' + p1)
#   f.write(image + '\n')
# f.close()
#----------------------------------------------------------

#solver
flags.DEFINE_string("train_dir", "models", "trained model save path")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
flags.DEFINE_string("imgs_list_path", "./train.txt", "images list file path")

flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

# flags.DEFINE_integer("num_epoch", 30, "train epoch num")
flags.DEFINE_integer("num_epoch", 1, "train epoch num")
flags.DEFINE_integer("batch_size", 32, "batch_size")

flags.DEFINE_float("learning_rate", 4e-4, "learning rate")

conf = flags.FLAGS
from solver import *

def main(_):
  solver = Solver()
  solver.train()

  # #-------------------------- 2021.10.18 NPU Modelarts文件传到OBS中-------------------------
  # # 解析输入参数train_url
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--train_url", type = str, default = "./output")
  # config = parser.parse_args()
  # # 在ModelArts容器创建训练输出目录
  # model_dir = "/cache/result"
  # os.makedirs(model_dir)
  # # 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
  # mox.file.copy_parallel(model_dir, config.train_url)
  # # -------------------------- 2021.10.18 NPU Modelarts文件传到OBS中-------------------------

if __name__ == '__main__':
  # EventLOG
  # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
  tf.app.run()
