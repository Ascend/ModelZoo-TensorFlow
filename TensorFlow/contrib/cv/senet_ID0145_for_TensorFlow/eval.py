# coding=utf-8
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
import argparse
from cifar10 import *

def parse_args():
    parser = argparse.ArgumentParser(description='MAIN')
    parser.add_argument('--total_epochs',type=int,default=10,help='total_epochs')
    parser.add_argument('--test_iteration',type=int,default=10,help='test_iteration')
    parser.add_argument('--data_path',type=str,help='data_path')
    args = parser.parse_args()
    return args

args = parse_args()


weight_decay = 0.0001
momentum = 0.9
init_learning_rate = 0.01
batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000
total_epochs = args.total_epochs
test_iteration = args.test_iteration
data_dir = args.data_path


def conv2d(input_tensor, filters, kernel_size, strides,  padding='SAME'):
    network = tf.layers.conv2d(inputs=input_tensor, use_bias=False, filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding)
    return network


def Relu(input_tensor):
    return tf.nn.relu(input_tensor)


def Sigmoid(input_tensor):
    return tf.nn.sigmoid(input_tensor)

def Global_Average_Pooling(input_tensor):
    return global_avg_pool(input_tensor, name='Global_avg_pooling')

def Fully_connected(x, units=class_num, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)


def bn(input_tensor):
    with slim.arg_scope([slim.batch_norm],
                        updates_collections=None,
                        decay=0.9,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True):
        return slim.batch_norm(inputs=input_tensor, is_training=False)

# def prelu(input_tensor):
#     return tl.layers.PReluLayer(input_tensor)

def resnet_layer(inputs,
                 filters=16,
                 kernel_size=(3, 3),
                 strides=1,
                 activation='relu',
                 batch_normalization=True):

    conv = tf.layers.Conv2D(filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs

    if batch_normalization:
        x = bn(x)
    if activation is not None:
        x = Relu(x)
    x = conv(x)
    # print(x.shape)

    squeeze = Global_Average_Pooling(x)
    excitation1 = Fully_connected(squeeze, units=int(filters / 16.0))
    excitation1 = Relu(excitation1)
    excitation2 = Fully_connected(excitation1, units=filters)
    excitation2 = Sigmoid(excitation2)
    excitation = tf.reshape(excitation2, shape=[-1, 1, 1, filters])
    scale = x * excitation
    # print(scale.shape)
    return scale


def seresnet_v2_eval(input_tensor, depth, num_classes=10, **kwargs):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # 开始模型定义。
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # v2 在将输入分离为两个路径前执行带 BN-ReLU 的 Conv2D 操作。
    x = resnet_layer(inputs=input_tensor,
                     filters=num_filters_in,
                     **kwargs)

    # 实例化残差单元的栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # 瓶颈残差单元
            y = resnet_layer(inputs=x,
                             filters=num_filters_in,
                             kernel_size=(1, 1),
                             strides=strides,
                             batch_normalization=batch_normalization,
                             **kwargs)
            # y = Squeeze_excitation_layer(y, num_filters_in, 16.0)
            y = resnet_layer(inputs=y,
                             filters=num_filters_in,
                             **kwargs)
            # y = Squeeze_excitation_layer(y, num_filters_in, 16.0)
            y = resnet_layer(inputs=y,
                             filters=num_filters_out,
                             kernel_size=(1, 1),
                             **kwargs)
            # y = Squeeze_excitation_layer(y, num_filters_out, 16.0)

            if res_block == 0:
                # 线性投影残差快捷键连接，以匹配更改的 dims
                # print(num_filters_out)
                # print(strides)
                x = resnet_layer(inputs=x,
                                 filters=num_filters_out,
                                 kernel_size=(1, 1),
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 **kwargs)
            # print(x.shape)
            x = x + y

        num_filters_in = num_filters_out

    # 在顶层添加分类器
    # v2 has BN-ReLU before Pooling
    x = bn(x)
    x = Relu(x)
    x = Global_Average_Pooling(x)
    x = slim.flatten(x)
    x = Fully_connected(x)
    return x
    
    
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


if __name__ == '__main__':
    import os
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    import tensorflow as tf
    from npu_bridge.npu_init import *
    import tf_slim as slim
    from tflearn.layers.conv import global_avg_pool
    
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = color_preprocessing(train_x, test_x)
    
    # image_size = 32, img_channels = 3, class_num = 10 in cifar10
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    #模型
    logits = seresnet_v2_eval(x, 110)
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    
    #loss—scale
    # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    # opt_tmp = npu_tf_optimizer(
    #      tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True))
    # optimizer = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
    
    optimizer = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True))
    train = optimizer.minimize(cost + l2_loss * weight_decay)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    
    
    
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
    
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    
        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
    
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            test_acc, test_loss, test_summary = Evaluate(sess)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()
    
            line = "epoch: %d/%d, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, test_loss, test_acc)
            print(line)
    
            with open('logs.txt', 'a') as f:
                f.write(line)
    
            # saver.save(sess=sess, save_path='model/senet110.ckpt')



