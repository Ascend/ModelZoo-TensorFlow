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
import os

import cv2
import moxing as mox
import numpy as np
import scipy.io as sio
import tensorflow as tf
from npu_bridge.npu_init import *

from tvnet import TVNet

flags = tf.app.flags
scale = 1
warp = 1
iteration = 50

# 设置npu服务器上的路径
data_path = '/home/ma-user/modelarts/inputs/data_url_0/'
output_path = '/home/ma-user/modelarts/outputs/train_url_0/'
print('data_url :' + data_path)
print('output_url :' + output_path)
# 从obs上获取模型
mox.file.copy_parallel('obs://cann-id0951/model/output/',output_path)

eval_data = os.listdir(output_path)  # 返回data_path下包含的文件或文件夹的名字的列表
print('输出目录下的文件：')
for name in eval_data:
    print(name)

# load image ,指定测试的图片对
img1 = cv2.imread(data_path + 'MPISintel_test/temple_2/frame_0001.png')
img2 = cv2.imread(data_path + 'MPISintel_test/temple_2/frame_0002.png')

h, w, c = img1.shape

# model construct
x1 = tf.placeholder(shape=[1, h, w, 3], dtype=tf.float32)
x2 = tf.placeholder(shape=[1, h, w, 3], dtype=tf.float32)
tvnet = TVNet()
u1, u2, rho = tvnet.tvnet_flow(x1, x2, max_scales=scale,
                               warps=warp,
                               max_iterations=iteration)
# init npu
# 变量初始化
init = tf.global_variables_initializer()
# 创建session
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
sess = tf.Session(config=config)

saver = tf.train.Saver()
saver = tf.train.import_meta_graph(output_path + 'ckpt_gpu_epe1/nn_model_gpu_epe.ckpt.meta')  # 加载模型结构
saver.restore(sess, tf.train.latest_checkpoint(output_path + 'ckpt_gpu_epe1/'))  # 只需要指定目录就可以恢复所有变量信息
all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  # 从一个集合中取出变量
# run model
u1_np, u2_np = sess.run([u1, u2], feed_dict={x1: img1[np.newaxis, ...], x2: img2[np.newaxis, ...]})

u1_np = np.squeeze(u1_np)
u2_np = np.squeeze(u2_np)
flow_mat = np.zeros([h, w, 2])
flow_mat[:, :, 0] = u1_np
flow_mat[:, :, 1] = u2_np



if not os.path.exists(output_path + 'result'):
    os.mkdir(output_path + 'result')
res_path = os.path.join(output_path + 'result', '1.mat')
sio.savemat(res_path, {'flow': flow_mat})
print("Extracting Flow finished!")

# 关闭sess
sess.close()

mox.file.copy_parallel(output_path, 'obs://cann-id0951/result/')
