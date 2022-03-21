#!/usr/bin/env python3
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
# -*- coding: UTF-8 -*-


# Reference:
# https://github.com/timctho/VNect-tensorflow
# https://github.com/EJShim/vnect_estimator


from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import pickle

# in tf.layers.conv2d: default_weight_name = kernel, default_bias_name = bias
# in tc.layers.conv2d: default_weight_name = weights, default_bias_name = biases


class VNect:
    def __init__(self):
        self.is_training = False
        self.input_holder = tf.placeholder(dtype=tf.float32, shape=(None, 368, 368, 3))
        self._build_network()

    def _build_network(self):
        # Conv
        self.conv1 = tc.layers.conv2d(self.input_holder, kernel_size=7, padding='same', num_outputs=64, stride=2,
                                      scope='conv1')
        self.pool1 = tc.layers.max_pool2d(self.conv1, kernel_size=3, padding='same', scope='pool1')

        # Residual block 2a
        self.res2a_branch1 = tc.layers.conv2d(self.pool1, kernel_size=1, padding='valid', num_outputs=256,
                                              activation_fn=None, scope='res2a_branch1')
        self.res2a_branch2a = tc.layers.conv2d(self.pool1, kernel_size=1, padding='valid', num_outputs=64,
                                               scope='res2a_branch2a')
        self.res2a_branch2b = tc.layers.conv2d(self.res2a_branch2a, kernel_size=3, padding='same', num_outputs=64,
                                               scope='res2a_branch2b')
        self.res2a_branch2c = tc.layers.conv2d(self.res2a_branch2b, kernel_size=1, padding='valid', num_outputs=256,
                                               activation_fn=None, scope='res2a_branch2c')
        self.res2a = tf.add(self.res2a_branch2c, self.res2a_branch1, name='res2a_add')
        self.res2a = tf.nn.relu(self.res2a, name='res2a')

        # Residual block 2b
        self.res2b_branch2a = tc.layers.conv2d(self.res2a, kernel_size=1, padding='valid', num_outputs=64,
                                               scope='res2b_branch2a')
        self.res2b_branch2b = tc.layers.conv2d(self.res2b_branch2a, kernel_size=3, padding='same', num_outputs=64,
                                               scope='res2b_branch2b')
        self.res2b_branch2c = tc.layers.conv2d(self.res2b_branch2b, kernel_size=1, padding='valid', num_outputs=256,
                                               activation_fn=None, scope='res2b_branch2c')
        self.res2b = tf.add(self.res2b_branch2c, self.res2a, name='res2b_add')
        self.res2b = tf.nn.relu(self.res2b, name='res2b')

        # Residual block 2c
        self.res2c_branch2a = tc.layers.conv2d(self.res2b, kernel_size=1, padding='valid', num_outputs=64,
                                               scope='res2c_branch2a')
        self.res2c_branch2b = tc.layers.conv2d(self.res2b_branch2a, kernel_size=3, padding='same', num_outputs=64,
                                               scope='res2c_branch2b')
        self.res2c_branch2c = tc.layers.conv2d(self.res2c_branch2b, kernel_size=1, padding='valid', num_outputs=256,
                                               activation_fn=None, scope='res2c_branch2c')
        self.res2c = tf.add(self.res2c_branch2c, self.res2b, name='res2c_add')
        self.res2c = tf.nn.relu(self.res2c, name='res2c')

        # Residual block 3a
        self.res3a_branch1 = tc.layers.conv2d(self.res2c, kernel_size=1, padding='valid', num_outputs=512,
                                              activation_fn=None, stride=2, scope='res3a_branch1')
        self.res3a_branch2a = tc.layers.conv2d(self.res2c, kernel_size=1, padding='valid', num_outputs=128, stride=2,
                                               scope='res3a_branch2a')
        self.res3a_branch2b = tc.layers.conv2d(self.res3a_branch2a, kernel_size=3, padding='same', num_outputs=128,
                                               scope='res3a_branch2b')
        self.res3a_branch2c = tc.layers.conv2d(self.res3a_branch2b, kernel_size=1, padding='valid', num_outputs=512,
                                               activation_fn=None, scope='res3a_branch2c')
        self.res3a = tf.add(self.res3a_branch2c, self.res3a_branch1, name='res3a_add')
        self.res3a = tf.nn.relu(self.res3a, name='res3a')

        # Residual block 3b
        self.res3b_branch2a = tc.layers.conv2d(self.res3a, kernel_size=1, padding='valid', num_outputs=128,
                                               scope='res3b_branch2a')
        self.res3b_branch2b = tc.layers.conv2d(self.res3b_branch2a, kernel_size=3, padding='same', num_outputs=128,
                                               scope='res3b_branch2b')
        self.res3b_branch2c = tc.layers.conv2d(self.res3b_branch2b, kernel_size=1, padding='valid', num_outputs=512,
                                               activation_fn=None, scope='res3b_branch2c')
        self.res3b = tf.add(self.res3b_branch2c, self.res3a, name='res3b_add')
        self.res3b = tf.nn.relu(self.res3b, name='res3b')

        # Residual block 3c
        self.res3c_branch2a = tc.layers.conv2d(self.res3b, kernel_size=1, padding='valid', num_outputs=128,
                                               scope='res3c_branch2a')
        self.res3c_branch2b = tc.layers.conv2d(self.res3c_branch2a, kernel_size=3, padding='same', num_outputs=128,
                                               scope='res3c_branch2b')
        self.res3c_branch2c = tc.layers.conv2d(self.res3c_branch2b, kernel_size=1, padding='valid', num_outputs=512,
                                               activation_fn=None, scope='res3c_branch2c')
        self.res3c = tf.add(self.res3c_branch2c, self.res3b, name='res3c_add')
        self.res3c = tf.nn.relu(self.res3c, name='res3c')

        # Residual block 3d
        self.res3d_branch2a = tc.layers.conv2d(self.res3c, kernel_size=1, padding='valid', num_outputs=128,
                                               scope='res3d_branch2a')
        self.res3d_branch2b = tc.layers.conv2d(self.res3d_branch2a, kernel_size=3, padding='same', num_outputs=128,
                                               scope='res3d_branch2b')
        self.res3d_branch2c = tc.layers.conv2d(self.res3d_branch2b, kernel_size=1, padding='valid', num_outputs=512,
                                               activation_fn=None, scope='res3d_branch2c')
        self.res3d = tf.add(self.res3d_branch2c, self.res3c, name='res3d_add')
        self.res3d = tf.nn.relu(self.res3d, name='res3d')

        # Residual block 4a
        self.res4a_branch1 = tc.layers.conv2d(self.res3d, kernel_size=1, padding='valid', num_outputs=1024,
                                              activation_fn=None, stride=2, scope='res4a_branch1')
        self.res4a_branch2a = tc.layers.conv2d(self.res3d, kernel_size=1, padding='valid', num_outputs=256, stride=2,
                                               scope='res4a_branch2a')
        self.res4a_branch2b = tc.layers.conv2d(self.res4a_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4a_branch2b')
        self.res4a_branch2c = tc.layers.conv2d(self.res4a_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4a_branch2c')
        self.res4a = tf.add(self.res4a_branch2c, self.res4a_branch1, name='res4a_add')
        self.res4a = tf.nn.relu(self.res4a, name='res4a')

        # Residual block 4b
        self.res4b_branch2a = tc.layers.conv2d(self.res4a, kernel_size=1, padding='valid', num_outputs=256,
                                               scope='res4b_branch2a')
        self.res4b_branch2b = tc.layers.conv2d(self.res4b_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4b_branch2b')
        self.res4b_branch2c = tc.layers.conv2d(self.res4b_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4b_branch2c')
        self.res4b = tf.add(self.res4b_branch2c, self.res4a, name='res4b_add')
        self.res4b = tf.nn.relu(self.res4b, name='res4b')

        # Residual block 4c
        self.res4c_branch2a = tc.layers.conv2d(self.res4b, kernel_size=1, padding='valid', num_outputs=256,
                                               scope='res4c_branch2a')
        self.res4c_branch2b = tc.layers.conv2d(self.res4c_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4c_branch2b')
        self.res4c_branch2c = tc.layers.conv2d(self.res4c_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4c_branch2c')
        self.res4c = tf.add(self.res4c_branch2c, self.res4b, name='res4c_add')
        self.res4c = tf.nn.relu(self.res4c, name='res4c')

        # Residual block 4d
        self.res4d_branch2a = tc.layers.conv2d(self.res4c, kernel_size=1, padding='valid', num_outputs=256,
                                               scope='res4d_branch2a')
        self.res4d_branch2b = tc.layers.conv2d(self.res4d_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4d_branch2b')
        self.res4d_branch2c = tc.layers.conv2d(self.res4d_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4d_branch2c')
        self.res4d = tf.add(self.res4d_branch2c, self.res4c, name='res4d_add')
        self.res4d = tf.nn.relu(self.res4d, name='res4d')

        # Residual block 4e
        self.res4e_branch2a = tc.layers.conv2d(self.res4d, kernel_size=1, padding='valid', num_outputs=256,
                                               scope='res4e_branch2a')
        self.res4e_branch2b = tc.layers.conv2d(self.res4e_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4e_branch2b')
        self.res4e_branch2c = tc.layers.conv2d(self.res4e_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4e_branch2c')
        self.res4e = tf.add(self.res4e_branch2c, self.res4d, name='res4e_add')
        self.res4e = tf.nn.relu(self.res4e, name='res4e')

        # Residual block 4f
        self.res4f_branch2a = tc.layers.conv2d(self.res4e, kernel_size=1, padding='valid', num_outputs=256,
                                               scope='res4f_branch2a')
        self.res4f_branch2b = tc.layers.conv2d(self.res4f_branch2a, kernel_size=3, padding='same', num_outputs=256,
                                               scope='res4f_branch2b')
        self.res4f_branch2c = tc.layers.conv2d(self.res4f_branch2b, kernel_size=1, padding='valid', num_outputs=1024,
                                               activation_fn=None, scope='res4f_branch2c')
        self.res4f = tf.add(self.res4f_branch2c, self.res4e, name='res4f_add')
        self.res4f = tf.nn.relu(self.res4f, name='res4f')

        # Residual block 5a
        self.res5a_branch2a_new = tc.layers.conv2d(self.res4f, kernel_size=1, padding='valid', num_outputs=512,
                                                   scope='res5a_branch2a_new')
        self.res5a_branch2b_new = tc.layers.conv2d(self.res5a_branch2a_new, kernel_size=3, padding='same',
                                                   num_outputs=512, scope='res5a_branch2b_new')
        self.res5a_branch2c_new = tc.layers.conv2d(self.res5a_branch2b_new, kernel_size=1, padding='valid',
                                                   num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
        self.res5a_branch1_new = tc.layers.conv2d(self.res4f, kernel_size=1, padding='valid', num_outputs=1024,
                                                  activation_fn=None, scope='res5a_branch1_new')
        self.res5a = tf.add(self.res5a_branch2c_new, self.res5a_branch1_new, name='res5a_add')
        self.res5a = tf.nn.relu(self.res5a, name='res5a')

        # Residual block 5b
        self.res5b_branch2a_new = tc.layers.conv2d(self.res5a, kernel_size=1, padding='valid', num_outputs=256,
                                                   scope='res5b_branch2a_new')
        self.res5b_branch2b_new = tc.layers.conv2d(self.res5b_branch2a_new, kernel_size=3, padding='same',
                                                   num_outputs=128, scope='res5b_branch2b_new')
        self.res5b_branch2c_new = tc.layers.conv2d(self.res5b_branch2b_new, kernel_size=1, padding='valid',
                                                   num_outputs=256, scope='res5b_branch2c_new')

        # Transpose Conv
        self.res5c_branch1a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=63,
                                                         activation=None, strides=2, padding='same', use_bias=False,
                                                         name='res5c_branch1a')
        self.res5c_branch2a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=128,
                                                         activation=None, strides=2, padding='same', use_bias=False,
                                                         name='res5c_branch2a')
        self.bn5c_branch2a = tc.layers.batch_norm(self.res5c_branch2a, scale=True, is_training=self.is_training,
                                                  scope='bn5c_branch2a')
        self.bn5c_branch2a = tf.nn.relu(self.bn5c_branch2a)

        self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z = tf.split(self.res5c_branch1a, num_or_size_splits=3,
                                                                              axis=3)
        self.res5c_branch1a_sqr = tf.multiply(self.res5c_branch1a, self.res5c_branch1a, name='res5c_branch1a_sqr')
        self.res5c_delta_x_sqr, self.res5c_delta_y_sqr, self.res5c_delta_z_sqr = tf.split(self.res5c_branch1a_sqr,
                                                                                          num_or_size_splits=3, axis=3)
        self.res5c_bone_length_sqr = tf.add(tf.add(self.res5c_delta_x_sqr, self.res5c_delta_y_sqr),
                                            self.res5c_delta_z_sqr)
        self.res5c_bone_length = tf.sqrt(self.res5c_bone_length_sqr)

        self.res5c_branch2a_feat = tf.concat(
            [self.bn5c_branch2a, self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z, self.res5c_bone_length],
            axis=3, name='res5c_branch2a_feat')

        self.res5c_branch2b = tc.layers.conv2d(self.res5c_branch2a_feat, kernel_size=3, padding='same', num_outputs=128,
                                               scope='res5c_branch2b')
        self.res5c_branch2c = tf.layers.conv2d(self.res5c_branch2b, kernel_size=1, padding='valid', filters=84,
                                               activation=None, use_bias=False, name='res5c_branch2c')
        # print(self.res5c_branch2c.get_shape())
        self.heatmap, self.x_heatmap, self.y_heatmap, self.z_heatmap = tf.split(self.res5c_branch2c,
                                                                                num_or_size_splits=4, axis=3)

    def load_weights(self, sess, params_file):
        # Read pretrained model file
        model_weights = pickle.load(open(params_file, 'rb'))

        # For each layer each var
        with tf.variable_scope('', reuse=True):
            for variable in tf.global_variables():
                var_name = variable.name.split(':')[0]
                # print(var_name)
                self.assign_weights_from_dict(var_name, model_weights, sess)

    @staticmethod
    def assign_weights_from_dict(var_name, model_weights, sess):
        with tf.variable_scope('', reuse=True):
            var_tf = tf.get_variable(var_name)
            print('assigning', var_tf)
            sess.run(tf.assign(var_tf, model_weights[var_name]))
            np.testing.assert_allclose(var_tf.eval(sess), model_weights[var_name])


if __name__ == '__main__':
    model = VNect()
    print('VNect building successfully.')
