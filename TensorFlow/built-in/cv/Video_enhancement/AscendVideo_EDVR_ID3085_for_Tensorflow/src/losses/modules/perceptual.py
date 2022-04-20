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
import numpy as np
import tensorflow as tf

from src.runner.common import name_space
from src.utils.logger import logger

from .vgg import VGG19_slim


def auto_download_pretrained(module='vgg', ckpt_path='./pretrained_modules'):
    """Automatically download pretrained models.

    Args:
        module: str, perceptual model name.
        ckpt_path: str, where to save the downloaded ckpt file.
    """
    import subprocess
    if module in ['vgg', 'vgg_19']:
        cmd0 = "wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz -O " + \
               os.path.join(ckpt_path, "vgg19.tar.gz")
        cmd0 += ";tar -xvf " + os.path.join(ckpt_path, "vgg19.tar.gz") + " -C " + ckpt_path + \
                "; rm " + os.path.join(ckpt_path, "vgg19.tar.gz")
    else:
        raise NotImplementedError

    subprocess.call(cmd0, shell=True)


def load_perceptual_module(sess, module_cfg):
    """Load perceptual module to the corresponding scope.
    
    Args:
        sess: tf.Session instance.
        module_cfg: yacs node, perceptual configuration.
    """
    ckpt_dir = module_cfg.get('ckpt_dir', './pretrained_modules')
    module = module_cfg.get('module', 'vgg_19')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_file = os.path.join(ckpt_dir, f'{module}.ckpt')
    if not os.path.exists(ckpt_file):
        logger.info('No pretrained module. Downloading ...')
        auto_download_pretrained(module, ckpt_dir)

    try:
        logger.info('Loading pretrained perceptual module ...')
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=module)
        restore = tf.train.Saver(var_list)
        restore.restore(sess, ckpt_file)
        logger.info('Load pretrained perceptual module success.')
    except Exception as e:
        logger.error('Failed to load pretrained perceptual model.')
        logger.info(e)


def build_perceptual_loss(generated, targets, module_cfg):
    """Calculate the perceptual loss given the configuration.

    Args:
        generated: tensor, the generated results.
        targets: tensor, the groundtruth.
        module_cfg: yacs node, perceptual configuration.

    Returns:
        scalar tensor, the perceptual loss.
    """
    module = module_cfg.get('module', 'vgg_19')

    # Convert to 4D shape.
    gen_shape = generated.get_shape().as_list()
    if len(gen_shape) == 5:
        generated = tf.reshape(generated, shape=(-1, *gen_shape[2:]))

    tar_shape = targets.get_shape().as_list()
    if len(tar_shape) == 5:
        targets = tf.reshape(targets, shape=(-1, *tar_shape[2:]))

    # Get the intermediate resuls given the layer names.
    if module == 'vgg_19':
        with tf.name_scope('vgg_19'):
            default_layer_labels = ['vgg_19/conv2/conv2_2',
                                    'vgg_19/conv3/conv3_4',
                                    'vgg_19/conv4/conv4_4',
                                    'vgg_19/conv5/conv5_4']
            default_layer_weights = [1., 1., 1., 1.]
            layer_labels = module_cfg.get('layers', default_layer_labels)
            layer_weights = module_cfg.get('layer_weights', default_layer_weights)
            gen_fm = VGG19_slim(generated, reuse=tf.AUTO_REUSE, deep_list=layer_labels)
            target_fm = VGG19_slim(targets, reuse=tf.AUTO_REUSE, deep_list=layer_labels)
    else:
        raise NotImplementedError

    # Compute the distance between the generated and groundtruth features.
    with tf.variable_scope('perceptual_loss'):
        loss = 0
        layer_n = len(layer_labels)

        for layer_i in range(layer_n):
            cur_diff = tf.reduce_sum(gen_fm[layer_labels[layer_i]] * target_fm[layer_labels[layer_i]], axis=[3])
            # cosine similarity, -1~1, 1 best
            cur_diff = 1.0 - tf.reduce_mean(cur_diff)  # 0 ~ 2, 0 best
            scaled_layer_loss = layer_weights[layer_i] * cur_diff
            loss += scaled_layer_loss

    return loss


def build_content_style_loss(generated, targets, module_cfg):
    """Calculate the perceptual style loss given the configuration.

    Args:
        generated: tensor, the generated results.
        targets: tensor, the groundtruth.
        module_cfg: yacs node, perceptual configuration.

    Returns:
        scalar tensor, the style loss.
    """
    module = module_cfg.get('module', 'vgg_19')

    gen_shape = generated.get_shape().as_list()
    if len(gen_shape) == 5:
        generated = tf.reshape(generated, shape=(-1, *gen_shape[2:]))

    tar_shape = targets.get_shape().as_list()
    if len(tar_shape) == 5:
        targets = tf.reshape(targets, shape=(-1, *tar_shape[2:]))

    if module == 'vgg_19':
        with tf.name_scope('vgg_19'):
            default_layer_labels = ['vgg_19/conv2/conv2_2',
                                    'vgg_19/conv3/conv3_4',
                                    'vgg_19/conv4/conv4_4',
                                    'vgg_19/conv5/conv5_4']
            default_layer_weights = [1., 1., 1., 1.]
            layer_labels = module_cfg.get('layers', default_layer_labels)
            layer_weights = module_cfg.get('layers_weights', default_layer_weights)
            print(layer_weights)
            gen_fm = VGG19_slim(generated, reuse=tf.AUTO_REUSE, deep_list=layer_labels, norm_flag=False)
            target_fm = VGG19_slim(targets, reuse=tf.AUTO_REUSE, deep_list=layer_labels, norm_flag=False)
    else:
        raise NotImplementedError

    with tf.variable_scope('perceptual_loss'):
        loss = 0
        layer_n = len(layer_labels)
        content_loss = 0
        style_loss = 0
        layers_content_weights = [0.008, 0.001, 0.03125, 40.0]
        layer_style_weights = [0.002, 0.000008, 0.03125, 10000.0]
        for layer_i in range(layer_n):
            f1 = gen_fm[layer_labels[layer_i]]
            f2 = target_fm[layer_labels[layer_i]]
            content_loss += layers_content_weights[layer_i] * tf.reduce_mean(tf.square(f1 / 10.0 - f2 / 10.0))
            if layer_i > 2:
                b,h,w,c = f1.shape
                f1T = tf.reshape(f1, (b, h*w, c))
                f2T = tf.reshape(f2, (b, h*w, c))
                f1G = tf.matmul(f1T, f1T, transpose_a=True)
                f2G = tf.matmul(f2T, f2T, transpose_a=True)
                norm = tf.cast(100.0 * h * w, tf.float32)
                style_loss += layer_style_weights[layer_i] * tf.reduce_mean(tf.square(f1G / norm - f2G / norm))
        loss = content_loss * 0.2 + style_loss * 0.08

    return loss