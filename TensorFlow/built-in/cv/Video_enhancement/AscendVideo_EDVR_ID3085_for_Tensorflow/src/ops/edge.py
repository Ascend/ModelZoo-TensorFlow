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

import numpy as np
import tensorflow as tf
from scipy import signal
import random


__all__ = ['tf_gaussian_blur', 'get_edges']


def gaussian_kernel(kernel_size, standard_dev):
    """Returns a 2D Gaussian kernel array with side length size and a sigma of 
    signal.

    Args:
        kernel_size: int.
        standard_dev: float, scalar of the kernel width.

    Returns:
        ndarray, a normalized np.ndarray of shape [kernel_size, kernel_size].
    """
    gkern1d = signal.gaussian(kernel_size, std=standard_dev).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return (gkern2d/gkern2d.sum())


def tf_gaussian_blur(x, kernel_size, standard_dev):
    """Apply gaussian blur to tensor using tf interface. Only works for RGB or 3-channel
    tensors.

    Args:
        x: tensor, 4D.
        kernel_size: int, blur kernel size.
        standard_dev: float.

    Returns:
        tensor, blured version of the input.
    """
    gau_k = gaussian_kernel(kernel_size, standard_dev)
    gau_0 = np.zeros_like(gau_k)
    gau_list = np.float32(  [
        [gau_k, gau_0, gau_0],
        [gau_0, gau_k, gau_0],
        [gau_0, gau_0, gau_k]]  ) # only works for RGB images!
    gau_wei = np.transpose(gau_list, [2,3,0,1])

    fix_gkern = tf.constant(gau_wei, dtype=tf.float32, shape=[kernel_size, kernel_size, 3, 3], name='gauss_blurWeights' )
    # shape [batch_size, crop_h, crop_w, 3]
    cur_data = tf.nn.conv2d(x, fix_gkern, strides=[1,1,1,1], padding="SAME", name='gauss_blur')
    return cur_data


def get_edges(x, method='sobel', use_default=False):
    """Get the edge map of a tensor x.

    Args:
        x: tensor, input feature map, whose number of channels can be larger than 3.
        method: str, which edge detector is used.
        use_default: boolean, whether to use tensorflow default sobel edge detector.
    
    Returns:
        tensor, the edge map of the input tensor.
    """
    if method == 'sobel' and use_default:
        edge = tf.image.sobel_edges(x)
        output_h, output_w = tf.split(edge, 2, axis=-1)
        output_h = tf.squeeze(output_h, axis=-1)
        output_w = tf.squeeze(output_w, axis=-1)
        edge_norm = tf.abs(output_h) * 0.5 + tf.abs(output_w) * 0.5
    elif method == 'sobel':
        # Blur before apply sobel operator.
        x = tf_gaussian_blur(x, 3, 1.2)
        x = tf.reduce_mean(x, axis=-1, keep_dims=True)
        kernel_h = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        kernel_w = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        x_pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        c = x.get_shape().as_list()[-1]
        conv_k_h = tf.constant(kernel_h, dtype=tf.float32, shape=(3, 3, 1, 1))
        conv_k_h = tf.tile(conv_k_h, (1, 1, c, 1))
        conv_k_w = tf.constant(kernel_w, dtype=tf.float32, shape=(3, 3, 1, 1))
        conv_k_w = tf.tile(conv_k_w, (1, 1, c, 1))
        output_h = tf.nn.depthwise_conv2d(x_pad, conv_k_h, strides=[1, 1, 1, 1], padding='VALID')
        output_w = tf.nn.depthwise_conv2d(x_pad, conv_k_w, strides=[1, 1, 1, 1], padding='VALID')
        edge_norm = tf.abs(output_h) * 0.5 + tf.abs(output_w) * 0.5
    elif method == 'laplacian':
        # Blur before apply edge operator.
        x = tf_gaussian_blur(x, 3, 1.2)
        x = tf.reduce_mean(x, axis=-1, keep_dims=True)
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        conv_k = tf.constant(kernel, dtype=tf.float32, shape=(3, 3, 1, 1))
        x_pad = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        output = tf.nn.depthwise_conv2d(x_pad, conv_k, strides=[1, 1, 1, 1], padding='VALID')
        edge_norm = tf.abs(output)
    else:
        raise NotImplementedError

    return edge_norm