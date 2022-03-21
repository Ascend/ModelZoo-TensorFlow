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
import numpy as np
import scipy.io as io


def get_variable(params, name):
    init = tf.constant_initializer(params, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=params.shape)
    return var

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def network(weights, image):
    net = {}
    current = image
    for layer_ind in range(weights['net'].shape[1]):
        layer_name = 'layer' + str(layer_ind)
        if layer_ind % 2 == 0:
            kernels = np.float32(weights['net'][0, layer_ind]['weights'][0, 0][0, 0])
            bias = np.float32(weights['net'][0, layer_ind]['weights'][0, 0][0, 1])
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=layer_name + "_w")
            bias = get_variable(bias.reshape(-1), name=layer_name + "_b")
            current = conv2d_basic(current, kernels, bias)
        else:
            current = tf.nn.relu(current, name=layer_name)
        net[layer_name] = current
    return net

class denoiser(object):
    """ Implements DAE objects with neural net parameters and functions. """
    def __init__(self, sess):
        self.sess = sess
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
        self.in_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")

        image_bgr = self.in_image[..., ::-1]
        weights = io.loadmat('/home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_sigma11.mat')
        with tf.variable_scope("dae", reuse=None):
            dae_net = network(weights=weights, image=image_bgr)
        print('====================dae================')
        print(dae_net)
        output_bgr = image_bgr + dae_net['layer' + str(weights['net'].shape[1] - 1)]
        self.output = output_bgr[..., ::-1]
        print('====================dae output=========')
        print(self.output)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # saver = tf.train.Saver()
        # saver.save(self.sess, "/cache/model/dmsp.cpkt")

    def denoise(self, noisy,gt,train=False):
        """ Implements the network forward pass to denoise an BGR image (in the range of 0 to 255) """
        if not train:
            return np.squeeze(self.sess.run(self.output, feed_dict={self.in_image: np.expand_dims(noisy, axis=0)}))
        else:
            gt_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="gt_image")
            loss = tf.reduce_sum(tf.pow(tf.subtract(self.in_image-self.output, gt_tensor),2.0))
            # update_op = self.opt.minimize(loss)
            loss_val = self.sess.run(loss, feed_dict={self.in_image: np.expand_dims(noisy, axis=0)
                ,gt_tensor: np.expand_dims(gt, axis=0)})
            return loss_val
