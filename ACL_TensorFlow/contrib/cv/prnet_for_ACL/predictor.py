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
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope
import numpy as np
from npu_bridge.estimator import npu_ops
import tensorflow.compat as compat


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
             scope=None):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
    with compat.forward_compatibility_horizon(2019, 5, 1):
        with tf.variable_scope(scope, 'resBlock'):
            shortcut = x
            if stride != 1 or x.get_shape()[3] != num_outputs:
                shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                      activation_fn=None, normalizer_fn=None, scope='shortcut')
            x = tcl.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding='SAME')
            x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding='SAME')
            x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

            x += shortcut
            x = normalizer_fn(x)
            x = activation_fn(x)
        return x


class resfcn256(object):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training=True):
        with compat.forward_compatibility_horizon(2019, 5, 1):
            with tf.variable_scope(self.name) as scope:
                with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                    with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                                   normalizer_fn=tcl.batch_norm,
                                   biases_initializer=None,
                                   padding='SAME',
                                   weights_regularizer=tcl.l2_regularizer(0.0002)):
                        size = 16
                        # x: s x s x 3
                        se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1)  # 256 x 256 x 16
                        se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
                        se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
                        se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
                        se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
                        se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
                        se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
                        se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
                        se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
                        se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
                        se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

                        with compat.forward_compatibility_horizon(2019, 5, 1):
                            pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
                            pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
                            pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                            pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
                            pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
                            pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                            pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
                            pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
                            pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
                            pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64

                            pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
                            pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
                            pd = tcl.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
                            pd = tcl.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16

                            pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                            pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
                            pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn=tf.nn.sigmoid)

                            return pos

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network(self.x, is_training=False)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    def restore(self, model_path):
        tf.train.Saver(self.network.vars).restore(self.sess, model_path)

    def predict(self, image):
        pos = self.sess.run(self.x_op, feed_dict={self.x: image[np.newaxis, :, :, :]})
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.MaxPos

# import tensorflow as tf
# import tensorflow.contrib.layers as tcl
# from tensorflow.contrib.framework import arg_scope
# import numpy as np
# from npu_bridge.estimator import npu_ops
# import tensorflow.compat as compat
#
#
# def resBlock(x, inchannels, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu,
#              normalizer_fn=tcl.batch_norm,
#              scope=None):
#     assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
#     with compat.forward_compatibility_horizon(2019, 5, 1):
#         with tf.variable_scope(scope, 'resBlock'):
#             shortcut = x
#             if stride != 1 or x.get_shape()[3] != num_outputs:
#                 shortcut = tf.nn.conv2d(input=shortcut, filter=[1, 1, inchannels, num_outputs],
#                                         strides=[1, stride, stride, 1], activation_fn=None, normalizer_fn=None)
#             x = tf.nn.conv2d(input=x, filter=[1, 1, inchannels, num_outputs / 2], strides=[1, 1, 1, 1], padding='SAME')
#             x = tf.nn.conv2d(input=x, filter=[kernel_size, kernel_size, num_outputs / 2, num_outputs / 2],
#                              strides=[1, stride, stride, 1], padding='SAME')
#             x = tf.nn.conv2d(input=x, filter=[1, 1, num_outputs / 2, num_outputs], strides=[1, 1, 1, 1],
#                              padding='SAME', normalizer_fn=None)
#
#             x += shortcut
#             x = normalizer_fn(x)
#             x = activation_fn(x)
#         return x
#
#
# class resfcn256(object):
#     def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
#         self.name = name
#         self.channel = channel
#         self.resolution_inp = resolution_inp
#         self.resolution_op = resolution_op
#
#     def __call__(self, x, is_training=True):
#         with compat.forward_compatibility_horizon(2019, 5, 1):
#             with tf.variable_scope(self.name) as scope:
#                 with arg_scope([tcl.batch_norm], training=is_training, scale=True):
#                     with arg_scope([tf.nn.conv2d, tf.nn.conv2d_transpose], activation_fn=tf.nn.relu,
#                                    normalizer_fn=tcl.batch_norm,
#                                    biases_initializer=None,
#                                    padding='SAME',
#                                    weights_regularizer=tcl.l2_regularizer(0.0002)):
#                         tf.nn.conv2d()
#                         tf.nn.l2_normalize
#                         size = 16
#                         # x: s x s x 3
#                         se = tf.nn.conv2d(input=x, filter=[4, 4, 3, size], strides=[1, 1, 1, 1])  # 256 x 256 x 16
#                         se = resBlock(se, inchannels=size, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
#                         se = resBlock(se, inchannels=size * 2, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
#                         se = resBlock(se, inchannels=size * 2, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
#                         se = resBlock(se, inchannels=size * 4, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
#                         se = resBlock(se, inchannels=size * 4, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
#                         se = resBlock(se, inchannels=size * 8, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
#                         se = resBlock(se, inchannels=size * 8, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
#                         se = resBlock(se, inchannels=size * 16, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
#                         se = resBlock(se, inchannels=size * 16, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
#                         se = resBlock(se, inchannels=size * 32, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512
#
#                         pd = tf.nn.conv2d_transpose(value=se, filter=[4, 4, size*32, size*32], output_shape=[16, 8, 8, size*32], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*16, size*32], output_shape=[16, 16, 16, size*16], strides=[1, 2, 2, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*16, size*16], output_shape=[16, 16, 16, size*16], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*16, size*16], output_shape=[16, 16, 16, size*16], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*8, size*16], output_shape=[16, 32, 32, size*8], strides=[1, 2, 2, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*8, size*8], output_shape=[16, 32, 32, size*8], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*8, size*8], output_shape=[16, 32, 32, size*8], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*4, size*8], output_shape=[16, 64, 64, size*4], strides=[1, 2, 2, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*4, size*4], output_shape=[16, 64, 64, size*4], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*4, size*4], output_shape=[16, 64, 64, size*4], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*2, size*4], output_shape=[16, 128, 128, size*2], strides=[1, 2, 2, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size*2, size*2], output_shape=[16, 128, 128, size*2], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size, size*2], output_shape=[16, 256, 256, size], strides=[1, 2, 2, 1])
#                         # pd = tcl.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, size, size], output_shape=[16, 256, 256, size], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, 3, size], output_shape=[16, 256, 256, 3], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
#                         pd = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, 3, 3], output_shape=[16, 256, 256, 3], strides=[1, 1, 1, 1])
#                         # pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
#                         pos = tf.nn.conv2d_transpose(value=pd, filter=[4, 4, 3, size], output_shape=[16, 256, 256, 3], strides=[1, 1, 1, 1], activation_fn=tf.nn.sigmoid)
#                         # pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn=tf.nn.sigmoid)
#
#                         return pos
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]
#
#
# class PosPrediction():
#     def __init__(self, resolution_inp=256, resolution_op=256):
#         # -- hyper settings
#         self.resolution_inp = resolution_inp
#         self.resolution_op = resolution_op
#         self.MaxPos = resolution_inp * 1.1
#
#         # network type
#         self.network = resfcn256(self.resolution_inp, self.resolution_op)
#
#         # net forward
#         self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
#         self.x_op = self.network(self.x, is_training=False)
#         self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
#
#     def restore(self, model_path):
#         tf.train.Saver(self.network.vars).restore(self.sess, model_path)
#
#     def predict(self, image):
#         pos = self.sess.run(self.x_op, feed_dict={self.x: image[np.newaxis, :, :, :]})
#         pos = np.squeeze(pos)
#         return pos * self.MaxPos
#
#     def predict_batch(self, images):
#         pos = self.sess.run(self.x_op, feed_dict={self.x: images})
#         return pos * self.MaxPos
