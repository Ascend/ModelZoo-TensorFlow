#
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
#
import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    # 第一二句为获取point的shape，bitchsize = 32，pointnum = 1024
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    # 第三句将输入的pointcloud拓展一维，变为32x1024x3x1的tensor，inputimage。
    input_image = tf.expand_dims(point_cloud, -1)

    # 第四、五、六句，则为搭建卷积层的过程，通过tf_util.conv2d函数实现。参考pointnet学习（八）tf_util.conv2d
    # 第一层卷积“tconv1”输出output（shpe[32，1024，1，64]）
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    # 第二层“tconv2”输出output（shpe[32，1024，1，128]）
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    # 第三层“tconv3”输出output（shpe[32，1024，1，1024]）
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    # 第七句为搭建maxpool层。因此“transform_net1”包括三个2d卷积层以及一个maxpoling层“tmaxpool”。
    # 输出为shape[32, 1, 1, 1024]的tensor
    # 参考pointnet tf_util.max_pool2d
    # 因为h, w, 都是1，所以可以将32个batch对应的每个input计算出来的1024个channel值取出来进行计算。
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    # 将net通过一个fullyconnect层进行计算。计算之后net为32，256的tensor
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    # 再后面的操作，则是对fullyconnect的输出，乘以一个weight，256，3 * k（k = 3）
    # 再加一个初始化为[1, 0, 0, 0, 1, 0, 0, 0, 1],shape为9的tensor
    # biases最后得到32，9的tensor
    # transform，再reshape成32，3，3的tensor，供后续预测对pointnet进行旋转，
    with tf.compat.v1.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.compat.v1.get_variable('weights', [256, 3*K],
                                  initializer=tf.compat.v1.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [3*K],
                                 initializer=tf.compat.v1.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases = biases + tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0]
    num_point = inputs.get_shape()[1]

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                  initializer=tf.compat.v1.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                 initializer=tf.compat.v1.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases = biases + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
