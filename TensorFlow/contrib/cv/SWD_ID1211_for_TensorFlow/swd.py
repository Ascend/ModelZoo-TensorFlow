#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function
from npu_bridge.npu_init import *
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.framework import get_variables
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops
import matplotlib.pyplot as plt
import imageio
import platform
import time

if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')


def toyNet(X):
    # Define network architecture
    with tf.variable_scope('Generator'):
        net = fully_connected(X, 15, activation_fn=tf.nn.relu)
        net = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net = fully_connected(net, 15, activation_fn=tf.nn.relu)
    with tf.variable_scope('Classifier1'):
        net1 = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net1 = fully_connected(net1, 15, activation_fn=tf.nn.relu)
        net1 = fully_connected(net1, 1, activation_fn=None)
        logits1 = tf.sigmoid(net1)
    with tf.variable_scope('Classifier2'):
        net2 = fully_connected(net, 15, activation_fn=tf.nn.relu)
        net2 = fully_connected(net2, 15, activation_fn=tf.nn.relu)
        net2 = fully_connected(net2, 1, activation_fn=None)
        logits2 = tf.sigmoid(net2)
    return logits1, logits2


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
        print("ssss")
        proj *= math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(proj), 0, keepdims=True))
        p1 = math_ops.matmul(p1, proj)
        p2 = math_ops.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = math_ops.reduce_mean(math_ops.square(p1 - p2))
    return math_ops.reduce_mean(wdist)


def discrepancy_mcd(out1, out2):
    return tf.reduce_mean(tf.abs(out1 - out2))


def load_data(data_path):
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    moon_data = np.load(data_path)
    x_s = moon_data['x_s']
    y_s = moon_data['y_s']
    x_t = moon_data['x_t']
    y_t = moon_data['y_t']
    return x_s, y_s, x_t, y_t


def generate_grid_point():
    x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
    y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    return xx, yy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="adapt_swd",
                        choices=["source_only", "adapt_mcd", "adapt_swd"])
    parser.add_argument('--data_path', type=str, default='./moon_data.npz')
    parser.add_argument('--steps', type=int, default=10001)
    opts = parser.parse_args()

    # Load data
    x_s, y_s, x_t, y_t = load_data(opts.data_path)

    # set random seed
    tf.set_random_seed(1234)

    # Define TF placeholders
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    X_target = tf.placeholder(tf.float32, shape=[None, 2])

    # Network definition
    with tf.variable_scope('toyNet'):
        logits1, logits2 = toyNet(X)
    with tf.variable_scope('toyNet', reuse=True):
        logits1_target, logits2_target = toyNet(X_target)

    # Cost functions
    eps = 1e-05
    cost1 = -tf.reduce_mean(Y * tf.log(logits1 + eps) + (1 - Y) * tf.log(1 - logits1 + eps))
    cost2 = -tf.reduce_mean(Y * tf.log(logits2 + eps) + (1 - Y) * tf.log(1 - logits2 + eps))
    loss_s = cost1 + cost2

    if opts.mode == 'adapt_swd':
        loss_dis = discrepancy_slice_wasserstein(logits1_target, logits2_target)
    else:
        loss_dis = discrepancy_mcd(logits1_target, logits2_target)

    # Setup optimizers
    variables_all = get_variables(scope='toyNet')
    variables_generator = get_variables(scope='toyNet' + '/Generator')
    variables_classifier1 = get_variables(scope='toyNet' + '/Classifier1')
    variables_classifier2 = get_variables(scope='toyNet' + '/Classifier2')

    optim_s = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s, var_list=variables_all)
    optim_dis1 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s - loss_dis, var_list=variables_classifier1)
    optim_dis2 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s - loss_dis, var_list=variables_classifier2)
    optim_dis3 = tf.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_dis, var_list=variables_generator)

    # Select predictions from C1
    predicted1 = tf.cast(logits1 > 0.5, dtype=tf.float32)

    # Generate grid points for visualization
    xx, yy = generate_grid_point()

    # For creating GIF purpose
    gif_images = []

    # Start session
    save_dir = 'model-npu/swd.ckpt'

    saver = tf.train.Saver()
    with tf.Session(config=npu_config_proto()) as sess:
        if opts.mode == 'source_only':
            print('-> Perform source only training. No adaptation.')
            train = optim_s
        else:
            print('-> Perform training with domain adaptation.')
            train = tf.group(optim_s, optim_dis1, optim_dis2, optim_dis3)

        # Initialize variables
        net_variables = tf.global_variables() + tf.local_variables()
        sess.run(tf.variables_initializer(net_variables))

        # Training
        for step in range(opts.steps):
            if step % 1000 == 0:
                print("Iteration: %d / %d" % (step, 10000))
                #### loss 计算 ########
                start_time = time.time()
                loss_s_step, loss_dis_step = sess.run([loss_s, loss_dis], feed_dict={X: x_s, Y: y_s, X_target: x_t})
                print("loss_s: %.5f , loss_dis: %.5f , step_time: %.4f" % (loss_s_step, loss_dis_step, (time.time()-start_time)))
                ####### acc 计算 #########
                logits1_step, logits1_target_step = sess.run(
                    [logits1, logits1_target], feed_dict={X: x_s, Y: y_s, X_target: x_t})
            
                pred1_step = (logits1_step > 0.5).astype(int)
                pred1_target_step = (logits1_target_step > 0.5).astype(int)
            
                acc_source = (pred1_step == y_s.reshape(-1, 1)).sum() / y_s.shape[0]
                acc_target = (pred1_target_step ==  y_t.reshape(-1, 1)).sum() / y_t.shape[0]
            
                print("source acc:%.2f, target acc:%.2f" % (acc_source, acc_target))
            
            # Forward and backward propagation
            _ = sess.run([train], feed_dict={X: x_s, Y: y_s, X_target: x_t}) 

        # Save GIF
        saver.save(sess, save_dir) 
        # imageio.mimsave(opts.mode + '.gif', gif_images, duration=0.8)
        print("[Finished]\n-> Please see the current folder for outputs.")

