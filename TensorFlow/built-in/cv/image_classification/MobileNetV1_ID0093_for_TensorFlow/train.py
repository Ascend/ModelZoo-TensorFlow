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
#!/usr/bin/env python
# -*- coding_utf-8 -*-
# ===========================
#   Author      : LZH
#   Time        : 2019-05-31
#   Language    : Python3
# ===========================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import cv2
import  os
import sys
import  argparse

import  tensorflow  as       tf
import  numpy       as       np
from    dataset     import   CifarData
from    mobilenet   import Mobilenet
from npu_bridge.npu_init import *

os.chdir(os.getcwd())

tf.reset_default_graph()

slim = tf.contrib.slim
def main(args):
    print_summary(args)
    x = tf.placeholder(dtype=tf.float32, shape=(None,3072),name='input_data')
    y_true = tf.placeholder(dtype=tf.int64, shape=(None),name='input_label' )
    is_train = tf.placeholder(dtype=tf.bool,name='is_train')
    global_step = tf.Variable(0, trainable=False)

    reshape_x = tf.reshape(x, [-1, 3, 32, 32])

    reshape_x = tf.transpose(reshape_x, perm=[0, 2, 3, 1])

    with tf.variable_scope("MobileNet"):

        #   the class Mobilenet will ouputs 3 variable, you should ignore the first two.
        #   you are expect to get the ouput from Mobilenet backbone
        #   and i use a avg_pooling and two fully connected layer, because i think the number of neural from 1024 to 10
        #   has a big gap, that will lead to information loss, so i use a transition FC layer to
        #   trainsit the information, and use leaky_relu to activate ouput from the first FC layer.

        _,_,output = Mobilenet(reshape_x, is_train).outputs
        avg_pooling = tf.nn.avg_pool(output,ksize=[1,7,7,1],strides=[1,1,1,1],padding="SAME",name="Avg_pooling")

        dense1 = tf.layers.dense(inputs=avg_pooling, units=512, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01), trainable=True,name="dense1")

        bn1 = tf.layers.batch_normalization(dense1, beta_initializer=tf.zeros_initializer(),
                                              gamma_initializer=tf.ones_initializer(),
                                              moving_mean_initializer=tf.zeros_initializer(),
                                              moving_variance_initializer=tf.ones_initializer(), training=is_train,
                                              name='bn1')
        relu1 = tf.nn.leaky_relu(bn1,0.1)

        dense2 = tf.layers.dense(inputs=relu1, units=10,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01), trainable=True,name="dense2")
        sqz = tf.squeeze(dense2,[1,2],name='sqz')

        prediction = tf.nn.softmax(sqz,name='prediction')


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=sqz))
    predict = tf.argmax(prediction, 1)

    correct_prediction = tf.equal(predict, y_true)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))


    moving_ave = tf.train.ExponentialMovingAverage(0.99).apply(tf.trainable_variables())

    train_filename = [os.path.join('./cifar', 'data_batch_%d' % i) for i in range(1, 6)]
    test_filename = [os.path.join('./cifar', 'test_batch')]
    

    saver = tf.train.Saver(max_to_keep=2)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.control_dependencies([moving_ave]):
            if (args.precision_mode=="allow_mix_precision"):
                optimizer = tf.train.AdamOptimizer(args.lr)
                loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                                       decr_every_n_nan_or_inf=2, decr_ratio=0.5)
                optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=False)
                train_op = optimizer.minimize(loss)
            else:
                train_op = tf.train.AdamOptimizer(args.lr).minimize(loss)


    with tf.Session() as sess:
        config_proto = tf.compat.v1.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
        config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess = tf.Session(config=npu_config_proto(config_proto=config_proto))
        sess.run(tf.global_variables_initializer())
        if args.load_pretrain == '1':
            saver.restore(sess,args.pretrain_path)
            print('Load pre_train model from: %s ' % args.pretrain_path)
        print('Start Training...')
        for epoch in range(1,args.epochs[0]+1):

            train_data = CifarData(train_filename, True)
            duration = 0
            for i in range(10000):

                batch_data, batch_labels, = train_data.next_batch(args.batch_size[0])
                start_time =time.time()
                loss_val, acc_val, _ ,predict1,yt= sess.run([loss, accuracy, train_op,predict,y_true],
                                                feed_dict={x: batch_data, y_true: batch_labels, is_train: True})
                duration += (time.time() - start_time)
                if (i + 1) % 1000 == 0:

                    print('[Train] Epoch: %d Step: %d, loss: %4.5f, acc: %.3f, perf: %4.5f' % (epoch, i + 1, loss_val, acc_val, duration))
                    duration = 0
                if (i + 1) % 10000 == 0:
                    test_data = CifarData(test_filename, False)
                    test_acc_sum = []
                    for j in range(100):
                        test_batch_data, test_batch_labels = test_data.next_batch(args.batch_size[0])
                        test_acc_val = sess.run([accuracy],
                                                feed_dict={x: test_batch_data, y_true: test_batch_labels, is_train: False})
                        test_acc_sum.append(test_acc_val)
                    test_acc = np.mean(test_acc_sum)
                    print('[Test ] acc: %4.5f' % (test_acc))
                    
            ckpt_file = "./ckpt/mobileNet_test_acc=%.4f.ckpt" % test_acc
            print('Save model to: %s \n' % ckpt_file)
            saver.save(sess, ckpt_file)

def print_summary(args):
    print('*'*30)
    print('precision_mode   :{}'.format(args.precision_mode))
    print('learning rate    : {}'.format(args.lr))
    print('Batch size       : {}'.format(args.batch_size[0]))
    print('Epoch            : {}'.format(args.epochs[0]))
    if args.load_pretrain == '1':
        print('load_pretrain    : YES')
        print('pretrain_path    : {}'.format(args.pretrain_path))
    else:
        print('load_pretrain    : No')
    print('*' * 30)

def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        type=float,
                        help='set lr',
                        default=1e-2)

    parser.add_argument('--batch_size',
                        type=int,
                        nargs='+',
                        help='Batch Size to train',
                        default=16)

    parser.add_argument('--epochs',
                        type=int,
                        nargs='+',
                        default=10,
                        help='Train Epochs')

    parser.add_argument('--load_pretrain',
                        type=str,
                        help='load_pretrain',
                        nargs='+',
                        default='1')

    parser.add_argument('--pretrain_path',
                        type=str,
                        nargs='+',
                        help='the path of pretrain model',
                        default='./ckpt/MobileNet.ckpt')
    parser.add_argument('--precision_mode',
                        type=str,
                        help='precision_mode set in NPU',
                        default='allow_fp32_to_fp16')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse(sys.argv[1:]))
