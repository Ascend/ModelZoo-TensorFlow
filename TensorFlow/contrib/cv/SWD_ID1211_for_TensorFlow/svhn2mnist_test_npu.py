#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

# In[1]:

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.test.gpu_device_name()
# use NPU


# # 华为ModelArt 相关
# * copy数据
# * 安装相关包

# In[2]:



# import moxing as mox
# mox.file.copy_parallel('obs://gather/MNIST_data', './datasets/mnist')
# mox.file.copy_parallel('obs://gather/SVHN', './datasets/svhn')


# In[3]:


# !pip install easydict
from npu_bridge.npu_init import *

# # 加载数据

# In[4]:


from datasets import LoadMNIST
from datasets import LoadSVHN
import numpy as np
def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    #     moon_data = np.load('moon_data.npz')
    #     x_s = moon_data['x_s']
    #     y_s = moon_data['y_s']
    #     x_t = moon_data['x_t']
    x_train_s, y_train_s = LoadSVHN.load(r'./datasets', subset='train')
    x_test_s, y_test_s = LoadSVHN.load(r'./datasets', subset='test')
#     x_train_s = LoadSVHN.rgb2gray(x_train_s)[:,:,:,np.newaxis]
#     x_test_s = LoadSVHN.rgb2gray(x_test_s)[:,:,:,np.newaxis]

    x_s = x_train_s
    y_s = y_train_s

    x_train_t, y_train_t, x_test_t, y_test_t = LoadMNIST.load(
        datadir="./datasets/mnist")
    x_train_t = LoadMNIST.imgs_resize(
        x_train_t)[:, :, :, np.newaxis].astype("float64")
    x_test_t = LoadMNIST.imgs_resize(
        x_test_t)[:, :, :, np.newaxis].astype("float64")
    x_train_t = np.concatenate([x_train_t, x_train_t, x_train_t], 3)
    x_test_t = np.concatenate([x_test_t, x_test_t, x_test_t], 3)

    x_t = x_train_t
    return x_s, y_s.reshape(-1, 1), x_test_s, y_test_s.reshape(-1, 1), x_t, y_train_t.reshape(-1, 1), x_test_t, y_test_t.reshape(-1, 1)

# opts = parser.parse_args()

# Load data
x_s, y_s, x_test_s, y_test_s, x_t, y_train_t, x_test_t, y_test_t = load_data()


# x_t.shape = (60000, 32, 32, 3)
# y_s.shape = (73257, 1)
# x_s.shape = (73257, 32, 32, 3)
# sparse_softmax_cross_entropy?


# # 定义模型结构

# In[5]:


#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d, flatten, batch_norm, dropout
from tensorflow.contrib.framework import get_variables
from tensorflow.contrib.losses import sparse_softmax_cross_entropy
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops
import matplotlib.pyplot as plt
import imageio
import platform


if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    
# from easydict import EasyDict as edict

# opts = edict()

# opts.mode = "adapt_mcd"
# opts.mode = "adapt_swd"
# opts.mode = "source_only"
# batch_size = 32



def toyNet(X):
    # Define network architecture
    with tf.variable_scope('Generator'):
        # net：svhn input [x,32,32,1] (convert rgb 2 gray , because MNIST's img is gray.)
        net = conv2d(inputs=X, num_outputs=64, kernel_size=[5, 5], stride=[
                     1, 1], padding='SAME', activation_fn=tf.nn.relu)  # 输出为[-1,32,32,64]
        net = batch_norm(net)
        net = max_pool2d(inputs=net, kernel_size=3, stride=3,
                         padding='SAME')  # 输出为[-1,11,11,64]

        net = conv2d(inputs=net, num_outputs=64, kernel_size=[5, 5], stride=[
                     1, 1], padding='SAME', activation_fn=tf.nn.relu)  # 输出为[-1,11,11,64]
        net = batch_norm(net)
        net = max_pool2d(inputs=net, kernel_size=3, stride=3,
                         padding='SAME')  # 输出为[-1,4,4,64]

        net = conv2d(inputs=net, num_outputs=128, kernel_size=[5, 5], stride=[
                     1, 1], padding='SAME', activation_fn=tf.nn.relu)  # 输出为[-1,4,4,128]
        net = batch_norm(net)
        
        net = flatten(net)  # [-1,2048]
#         net = dropout(net)
    with tf.variable_scope('Classifier1'):
        net1 = batch_norm(fully_connected(net, 2048, activation_fn=tf.nn.relu))
        net1 = batch_norm(fully_connected(net1, 2048, activation_fn=tf.nn.relu))
        net1 = fully_connected(net1, 10, activation_fn=None)
#         net1 = fully_connected(net, 10, activation_fn=None)

        # logits1 = tf.sof(net1)
    with tf.variable_scope('Classifier2'):
        net2 = batch_norm(fully_connected(net, 2048, activation_fn=tf.nn.relu))
        net2 = batch_norm(fully_connected(net2, 2048, activation_fn=tf.nn.relu))
        net2 = fully_connected(net2, 10, activation_fn=None)
#         net2 = fully_connected(net, 10, activation_fn=None)

    return net1, net2


def sort_rows(matrix, num_rows):
    matrix_T = array_ops.transpose(matrix, [1, 0])
    sorted_matrix_T = nn_ops.top_k(matrix_T, num_rows)[0]
    return array_ops.transpose(sorted_matrix_T, [1, 0])


def discrepancy_slice_wasserstein(p1, p2):
    s = array_ops.shape(p1)
    if p1.get_shape().as_list()[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        # 对于一维以上的数据，执行到一维的多重随机投影
        proj = random_ops.random_normal([array_ops.shape(p1)[1], 128])
        proj *= math_ops.rsqrt(math_ops.reduce_sum(
            math_ops.square(proj), 0, keepdims=True))  # keep_dims -> keepdims
        p1 = math_ops.matmul(p1, proj)
        p2 = math_ops.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = math_ops.reduce_mean(math_ops.square(p1 - p2))
    return math_ops.reduce_mean(wdist)


def discrepancy_mcd(out1, out2):
    return tf.reduce_mean(tf.abs(out1 - out2))

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default="adapt_swd",
                    choices=["source_only", "adapt_mcd", "adapt_swd"])
opts = parser.parse_args()

# set random seed
tf.set_random_seed(1234)

# Define TF placeholders
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.int32, shape=[None, 1])
X_target = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

# Network definition
with tf.variable_scope('toyNet'):
    logits1, logits2 = toyNet(X)
with tf.variable_scope('toyNet', reuse=True):
    logits1_target, logits2_target = toyNet(X_target)

# Cost functions
cost1 = sparse_softmax_cross_entropy(logits=logits1, labels=Y)
cost2 = sparse_softmax_cross_entropy(logits=logits2, labels=Y)
loss_s = cost1 + cost2

if opts.mode == 'adapt_swd':
    loss_dis = discrepancy_slice_wasserstein(
        logits1_target, logits2_target)
else:
    loss_dis = discrepancy_mcd(logits1_target, logits2_target)

# Setup optimizers
variables_all = get_variables(scope='toyNet')
variables_generator = get_variables(scope='toyNet' + '/Generator')
variables_classifier1 = get_variables(scope='toyNet' + '/Classifier1')
variables_classifier2 = get_variables(scope='toyNet' + '/Classifier2')

optim_s = tf.train.GradientDescentOptimizer(learning_rate=0.05).    minimize(loss_s, var_list=variables_all)
optim_dis1 = tf.train.GradientDescentOptimizer(learning_rate=0.005).    minimize(loss_s - loss_dis, var_list=variables_classifier1)
optim_dis2 = tf.train.GradientDescentOptimizer(learning_rate=0.005).    minimize(loss_s - loss_dis, var_list=variables_classifier2)
optim_dis3 = tf.train.GradientDescentOptimizer(learning_rate=0.5).    minimize(loss_dis, var_list=variables_generator)

# optim_s = tf.train.AdamOptimizer(learning_rate=0.005).\
#     minimize(loss_s, var_list=variables_all)
# optim_dis1 = tf.train.AdamOptimizer(learning_rate=0.005).\
#     minimize(loss_s - loss_dis, var_list=variables_classifier1)
# optim_dis2 = tf.train.AdamOptimizer(learning_rate=0.005).\
#     minimize(loss_s - loss_dis, var_list=variables_classifier2)
# optim_dis3 = tf.train.AdamOptimizer(learning_rate=0.005).\
#     minimize(loss_dis, var_list=variables_generator)

# Select predictions from C1
predicted1 = tf.math.argmax(input = logits1, axis=1)


# # 训练模型

# In[6]:


# Start session
batch_size = 1000
# with tf.Session() as sess:
with tf.Session(config=npu_config_proto()) as sess:
    if opts.mode == 'source_only':
        print('-> Perform source only training. No adaptation.')
        train = optim_s
    else:
        print('-> Perform training with domain adaptation.')
        train = tf.group(optim_s, optim_dis1, optim_dis2, optim_dis3)
    saver = tf.train.Saver()
    max_acc = 0
    # Initialize variables
    net_variables = tf.global_variables() + tf.local_variables()
    sess.run(tf.variables_initializer(net_variables))
    
#     saver = tf.train.Saver()
    save_dir = f"model-{opts.mode}/graph.ckpt"
    saver.restore(sess, save_dir)
    # Training
    for step in range(1):
        # Forward and backward propagation
#         batch_size = x_t.shape[0]

                
        if step % 1 == 0:
            # print(f"Iteration: {step} / 1, loss_s is {loss_s_tmp}, loss_dis is {loss_dis_tmp}" )
#             print(loss_s_tmp)
            predicted_y_test_t = []
            predicted_y_test_s = []
    
            for j in range(0, x_test_t.shape[0], batch_size):
                s, e = j, j + batch_size

                batch_x_test_t = x_test_t[s:e]
                batch_x_test_s = x_test_s[s:e]
                batch_predicted_y_test_t = sess.run(predicted1, feed_dict={
                             X: batch_x_test_t})
                batch_predicted_y_test_s = sess.run(predicted1, feed_dict={
                             X: batch_x_test_s})
                predicted_y_test_t.append(batch_predicted_y_test_t)
                predicted_y_test_s.append(batch_predicted_y_test_s)
                
#                 print(batch_x_test_t.shape)
#                 print(batch_predicted_y_test_t.shape)
            predicted_y_test_t = np.array(predicted_y_test_t).reshape(-1, 1)
            predicted_y_test_s = np.array(predicted_y_test_s).reshape(-1, 1)
            
            acc_test_t = (predicted_y_test_t == y_test_t).sum() / y_test_t.shape[0]
            acc_test_s = (predicted_y_test_s == y_test_s[:10000]).sum() / 10000
            if acc_test_t > max_acc:
                max_acc = acc_test_t
                saver.save(sess, f'model-{opts.mode}/graph.ckpt')
            print(f"MNIST target accuracy is {acc_test_t},SVHN source acc is {acc_test_s}")
    # Save GIF
#     imageio.mimsave(opts.mode + '.gif', gif_images, duration=0.8)
    print("[Finished]\n-> Please see the current folder for outputs.")





