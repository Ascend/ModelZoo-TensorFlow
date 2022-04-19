# Copyright 2022 Huawei Technologies Co., Ltd
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

def spectral_norm(w, iteration=1):
    """Spectral normalization of kernels.

    Borrowed from https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py

    Args:
        w: tensor, conv/linear layer kernel.
        iteration: int, number of power iteration.

    Returns:
        A normalized kernel tensor.
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("spectral_norm_u", [1, w_shape[-1]], 
                        initializer=tf.truncated_normal_initializer(), 
                        trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        # power iteration
        # Usually iteration = 1 will be enough
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm