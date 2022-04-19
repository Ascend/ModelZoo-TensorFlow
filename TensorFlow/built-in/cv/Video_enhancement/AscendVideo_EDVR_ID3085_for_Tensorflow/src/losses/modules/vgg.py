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

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

VGG_MEAN = [123.68, 116.78, 103.94]


def vgg_19(inputs,
           scope='vgg_19',
           reuse=False):
    """VGG19 model.
    Borrowed from https://github.com/thunil/TecoGAN/blob/master/lib/ops.py#L287
    Changed from the Oxford Net VGG 19-Layers version E Example.
    Note: Only offer features from conv1 until relu54, classification part is removed
    
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      scope: Optional scope for the variables.
    
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points


def VGG19_slim(input_fm, reuse, deep_list=None, norm_flag=True):
    """Get the VGG19 features given the fm name.
    Borrowed from https://github.com/thunil/TecoGAN/blob/master/lib/Teco.py#L5

    Args:
        input_fm: tensor, input feature map.
        reuse: boolean, whether to reuse the scope variables.
        deep_list: list[str], which features are to extract and used for calculation.
        norm_flag: boolean, whether to normalize the feature map with Frobenius-norm.
    """
    # deep_list, define the feature to extract
    input_img_ab = input_fm * 255.0 - tf.constant(VGG_MEAN)
    # model:
    _, output = vgg_19(input_img_ab, reuse=reuse)
    # feature maps:
    results = {}
    with tf.name_scope('vgg_norm'):
        for key in output:
            if (deep_list is None) or (key in deep_list):
                orig_deep_feature = tf.cast(output[key], tf.float32)
                if norm_flag:
                    orig_len = tf.sqrt(tf.reduce_sum(tf.square(orig_deep_feature), axis=[3], keepdims=True)+1e-12)
                    results[key] = orig_deep_feature / orig_len
                else:
                    results[key] = orig_deep_feature
    return results
