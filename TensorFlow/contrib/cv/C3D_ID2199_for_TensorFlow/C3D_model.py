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
import tensorflow.contrib.slim as slim


def C3D(input, num_classes, keep_pro=0.5):
    with tf.variable_scope('C3D'):
        net = tf.layers.conv3d(input, 64, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv1')
        net = tf.layers.max_pooling3d(net, pool_size=[1, 2, 2], strides=[1, 2, 2], padding='SAME', name='max_pool1')
        net = tf.layers.conv3d(net, 128, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv2')
        net = tf.layers.max_pooling3d(net, pool_size=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='max_pool2')
        net = tf.layers.conv3d(net, 256, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv3')
        net = tf.layers.conv3d(net, 256, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv4')
        net = tf.layers.max_pooling3d(net, pool_size=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='max_pool3')
        net = tf.layers.conv3d(net, 512, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv5')
        net = tf.layers.conv3d(net, 512, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv6')
        net = tf.layers.max_pooling3d(net, pool_size=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='max_pool4')
        net = tf.layers.conv3d(net, 512, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv7')
        net = tf.layers.conv3d(net, 512, kernel_size=[3, 3, 3], strides=[1, 1, 1], activation=tf.nn.relu,
                               padding='SAME', name='conv8')
        net = tf.layers.max_pooling3d(net, pool_size=[2, 2, 2], strides=[2, 2, 2], padding='SAME', name='max_pool5')

        net = tf.reshape(net, [-1, 512 * 4 * 4])
        net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc6')
        net = slim.dropout(net, keep_pro, scope='dropout1')
        net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc7')
        net = slim.dropout(net, keep_pro, scope='dropout2')
        out = slim.fully_connected(net, num_classes, weights_regularizer=slim.l2_regularizer(0.0005),
                                   activation_fn=None, scope='out')

        return out
