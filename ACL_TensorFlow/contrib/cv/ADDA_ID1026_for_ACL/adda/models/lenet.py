"""
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
"""
from collections import OrderedDict
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn

@register_model_fn('lenet')
def lenet(inputs, scope='lenet', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(slim.arg_scope([slim.conv2d], padding='VALID'))
            net = slim.conv2d(net, 20, 5, scope='conv1')
            layers['conv1'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            layers['pool1'] = net
            net = slim.conv2d(net, 50, 5, scope='conv2')
            layers['conv2'] = net
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            layers['pool2'] = net
            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 500, scope='fc3')
            layers['fc3'] = net
            net = slim.fully_connected(net, 10, activation_fn=None, scope='fc4')
            layers['fc4'] = net
    return net, layers
lenet.default_image_size = 28
lenet.num_channels = 1
lenet.mean = None
lenet.bgr = False

