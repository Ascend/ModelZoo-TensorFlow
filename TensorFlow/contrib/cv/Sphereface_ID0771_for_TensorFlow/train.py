# -*- coding: utf-8 -*-
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
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import model
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
from absl import flags,app
#os.environ['CUDA_VISIBLE_DEVICES']='1'
#os.environ['EXPERIMENTAL_DYNAMIC_PARTITION'] = "1"
"""
@platform: vim
@author:   YunYang1994
Created on sunday July 15  16:25:45 2018

    -->  -->
       ==
##########################################################################

This is quick re-implementation of asoftmax loss proposed in this paper:
    'SphereFace: Deep Hypersphere Embedding for Face Recognition. '
see https://arxiv.org/pdf/1704.08063.pdf
if you have any questions, please contact with me, I am very happy to
discuss them with you, my email is 'dreameryangyun@sjtu.edu.cn'

#########################################################################
"""
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_path', default='./MNIST_data',
    help=('Directory to store dataset'))

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'train_batchs', default=40,
    help=('Batch number for training.'))

flags.DEFINE_integer(
    'test_batch_size', default=256,
    help=('Batch size for testing.'))

flags.DEFINE_integer(
    'test_batchs', default=20,
    help=('Batch number for testing.'))

flags.DEFINE_integer(
    'epochs', default=40,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'lr', default=0.001,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_string(
    'model_path', default='./ckpt/',
    help=('Directory to store model data'))

flags.DEFINE_string(
    'train_url', default='./output',
    help=('Directory to store in OBS'))

# prepare mnist data


# define training parameters


def train(argv):
    dataset_path = FLAGS.dataset_path
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    train_batchs = FLAGS.train_batchs # the number of batchs per epoch
    test_batch_size = FLAGS.test_batch_size
    test_batchs  = FLAGS.test_batchs
    model_path = FLAGS.model_path
    train_url = FLAGS.train_url
    # define input placeholder for network
    mnist = input_data.read_data_sets(dataset_path, one_hot=False, reshape=False)
    images = tf.placeholder(tf.float32, shape=[batch_size,28,28,1], name='input')
    labels = tf.placeholder(tf.int64, [batch_size,])
    test_images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_x')
    test_labels = tf.placeholder(tf.int64, [None, ], name='input_y')
    global_step = tf.Variable(0, trainable=False)
    add_step_op = tf.assign_add(global_step, tf.constant(1))
    # about network
    with tf.variable_scope('sphere20', reuse=tf.AUTO_REUSE):
        #train network
        network = model.Model(images, labels)
        accuracy = network.accuracy
        loss = network.loss
        #test network
    with tf.variable_scope('sphere20', reuse=tf.AUTO_REUSE):
        test_network = model.Model(test_images, test_labels)
        test_accuracy = test_network.accuracy
        test_loss = test_network.loss

    # define optimizer and learning rate
    decay_lr = tf.train.exponential_decay(lr, global_step, 500, 0.9)
    optimizer = tf.train.AdamOptimizer(decay_lr)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=npu_config_proto(config_proto=config))
    sess.run(tf.global_variables_initializer())
    # training process

    for epoch in range(epochs):
        train_acc = 0.
        for batch in range(train_batchs):

            batch_images, batch_labels = mnist.train.next_batch(batch_size)
            feed_dict = {images:batch_images, labels:batch_labels}
            t_start = time.time()
            _, _, batch_loss, batch_acc = sess.run([train_op, add_step_op, loss, accuracy], feed_dict)
            t_end = time.time()
            # print(batch_acc)
            train_acc += batch_acc

            perf = t_end - t_start
            fps = batch_size / perf
            print('epoch: {} step: {} perf: {:.2f} FPS: {:.2f} train_loss: {:.6f}'.format(epoch + 1, batch+1, perf, fps,
                                                                                          batch_loss))
        train_acc /= train_batchs
        print("epoch %2d---------------------------train_accuracy:%.4f" %(epoch+1, train_acc))
        #visualize(embeddings, nlabels, epoch, train_acc, picname="./image/%d/%d.jpg"%(loss_type, epoch))

    # testing process


    test_acc = 0.
    for batch in range(test_batchs):
        batch_images, batch_labels = mnist.test.next_batch(test_batch_size)
        feed_dict = {test_images:batch_images, test_labels:batch_labels}
        batch_loss, batch_acc = sess.run([test_loss, test_accuracy], feed_dict)
        test_acc += batch_acc

    test_acc /= test_batchs
    print("test_accuracy: %.4f" %test_acc)
    saver.save(sess, model_path + 'sphereface.ckpt')
    tf.io.write_graph(sess.graph, model_path, 'graph.pbtxt', as_text=True)
    sess.close()


if __name__ == "__main__":
    app.run(train)




