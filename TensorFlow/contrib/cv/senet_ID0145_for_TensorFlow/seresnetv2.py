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

# coding=utf-8

# from tensorflow import keras
# from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape
# from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, multiply
# from tensorflow.keras.regularizers import l2

import tensorflow as tf
import tf_slim as slim
from tflearn.layers.conv import global_avg_pool
from cifar10 import *

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
        # return tf.cond(training,
         #             lambda: slim.batch_norm(inputs=input_tensor, is_training=training, reuse=None),
         #             lambda: slim.batch_norm(inputs=input_tensor, is_training=training, reuse=True))

         return slim.batch_norm(inputs=input_tensor, is_training=True)


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


def seresnet_v2(input_tensor, depth, num_classes=10, **kwargs):

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
            
            y = resnet_layer(inputs=y,
                             filters=num_filters_in,
                             **kwargs)
           
            y = resnet_layer(inputs=y,
                             filters=num_filters_out,
                             kernel_size=(1, 1),
                             **kwargs)
            

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


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # resnet = ResNet(version=2,
    #                 input_shape=[32, 32, 3],
    #                 depth=56,
    #                 num_classes=10)
    # resnet.summary()
