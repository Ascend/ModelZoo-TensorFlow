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
import datetime
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from npu_bridge.npu_init import *
import random
from tvnet import TVNet, batch_size
import argparse
import sys

flags = tf.app.flags
scale = 1
warp = 1
iteration = 50
print('TVNet Params:\n scale: %d\n warp: %d\n iteration: %d' \
      % (scale, warp, iteration))


def get_config(args):
    parser = argparse.ArgumentParser(description='Experiment parameters')
    parser.add_argument('--data_path', default='./dataset', help='training input data path.')
    parser.add_argument('--output_path', default='./output', help='prepocess result path.')
    parser.add_argument('--steps', default='10000', help='train steps.')
    parsed_args, unknown_args = parser.parse_known_args(args)
    return parsed_args


def readflo(file_name):  # 读取光流文件
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file' % (w, h)
            data = np.array(np.fromfile(f, np.float32, count=2 * int(w) * int(h)))
            # Reshape data into 3D array (columns, rows, bands)
            # data2D = np.ndarray.reshape(data, (w, h, 2))
            data2D = data.reshape(int(h), int(w), 2)
            return data2D


# other_data = os.listdir("./other_data/")

def loadSintelData(data_url):  # 加载训练数据
    data_path = os.path.join(data_url, "MPISintel_train/")
    # print(data_url + '\n')
    eval_data = os.listdir(data_path)  # 返回data_path下包含的文件或文件夹的名字的列表
    # for name in eval_data:
    #     print(name + '/n')
    img1 = np.zeros((batch_size, 436, 1024, 3))
    img2 = np.zeros((batch_size, 436, 1024, 3))
    label = np.zeros((batch_size, 436, 1024, 2))
    lod_folder = random.sample(eval_data, 1)[0]
    train_dir = data_path + lod_folder
    for j in range(batch_size):
        i = random.randint(1, 49)
        img1[j, :] = cv2.imread(train_dir + "/frame_" + str(i).zfill(4) + ".png")
        img2[j, :] = cv2.imread(train_dir + "/frame_" + str(i + 1).zfill(4) + ".png")
        label[j, :] = readflo(train_dir + "/frame_" + str(i).zfill(4) + ".flo")
        # print("===>>>Flow File: "+ train_dir + "/frame_" + str(i).zfill(4) + ".flo")
    return img1, img2, label


def calculate_epe(pr_u1, pr_u2, gt_u):
    pr_u1 = tf.squeeze(pr_u1)
    pr_u2 = tf.squeeze(pr_u2)
    return tf.reduce_mean(tf.sqrt(tf.square(pr_u1 - gt_u[:, :, 0]) + tf.square(pr_u2 - gt_u[:, :, 1])))


def calculate_loss(u1, u2, y):
    loss = 0
    for j in range(batch_size):
        y_1 = u1[j, :]

        y_2 = u2[j, :]

        gt = y[j, :]

        loss += calculate_epe(y_1, y_2, gt)

    return loss / batch_size


x1 = tf.placeholder(shape=[batch_size, 436, 1024, 3], dtype=tf.float32)  # 函数作为一种占位符用于定义过程，可以理解为形参，在执行的时候再赋具体的值
x2 = tf.placeholder(shape=[batch_size, 436, 1024, 3], dtype=tf.float32)
y = tf.placeholder(shape=[batch_size, 436, 1024, 2], dtype=tf.float32)
tf.summary.image('input', [x1, x2])  # 形成一张名为input的图像

loss_list = []

tvnet = TVNet()  # 初始化TVnet类

u1_p, u2_p, rho = tvnet.tvnet_flow(x1, x2, max_scales=scale,
                                   warps=warp,
                                   max_iterations=iteration)

loss = calculate_loss(u1_p, u2_p, y)  # 计算loss
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)  # 设置优化器

# 设置npu服务器上的路径
args = get_config(sys.argv[1:])
max_steps = int(args.steps)
print('data_url :' + args.data_path)
print('output_url :' + args.output_path)
print('steps:' + args.steps)
eval_data = os.listdir(args.data_path)  # 返回data_url下包含的文件或文件夹的名字的列表
for name in eval_data:
    print(name)

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
sess.run(init)

saver = tf.train.Saver(tf.global_variables())  # 模型的保存和加载

start = datetime.datetime.now()
for step in range(max_steps):  # 开始训练
    start_time = time.time()
    img1, img2, label = loadSintelData(args.data_path)
    # img1, img2 = loadData(batch_size)
    _, loss_value = sess.run([train_op, loss], feed_dict={x1: img1, x2: img2, y: label})  # 带入具体的值
    duration = time.time() - start_time
    loss_list.append(loss_value)
    if step % 5 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
# Total time
# 按照格式输出单步训练的时间
end = datetime.datetime.now()
timefortrain = (end - start).total_seconds()
cost_time = timefortrain / max_steps
print("sec/step : {}".format(cost_time))
print("use second")
print(timefortrain)
strtime = '%dh%dm%ds' % (timefortrain / 3600, timefortrain % 3600 / 60, timefortrain % 3600 % 60)
print("===>>>Total train Time: " + strtime)  # 输出训练时间

ckpt_path = os.path.join(args.output_path, 'ckpt_gpu_epe1')  # 模型最终保存路径：./output/ckpt_gpu_epe1/
if not os.path.exists(ckpt_path):  # 判断模型保存的路径是否存在
    os.mkdir(ckpt_path)
checkpoint_path = os.path.join(ckpt_path, "nn_model_gpu_epe.ckpt")
print("===>>>checkpoint_path: " + checkpoint_path)
saver.save(sess, checkpoint_path)  # 保存模型
# 关闭sess
sess.close()

loss_path = os.path.join(args.output_path, "loss_Sintel_gpu.log")  # loss_log最终保存路径：./output
print("===>>>loss_path: " + loss_path)

# Average and minal value of loss list
avg_loss = np.mean(loss_list)
min_loss = np.min(loss_list)
print("Average epeLoss: " + str(avg_loss) + "; Minimam Loss: " + str(min_loss))

#  开始写入loss
loss_list1 = []
file = open(loss_path, 'w')
file.write("Total_train_Time: " + strtime)
file.write("Average_epeLoss: " + str(avg_loss) + "; Minimam_Loss: " + str(min_loss))
for i in range(len(loss_list)):
    loss_list1.append(np.mean(loss_list[0:i]))
    file.write(str(loss_list[i]))
    file.write("\n")
file.close()
loss_path1 = os.path.join(args.output_path, "epeloss1_Sintel_gpu.log")
print("===>>>loss1_path: " + loss_path1)




