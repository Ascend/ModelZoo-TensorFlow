# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
import tensorflow.contrib.slim as slim


def resnet_arg_scope(
        is_training=True,
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        activation_fn=tf.nn.relu,
        use_batch_norm=True,
        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': batch_norm_updates_collections,
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.dropout], is_training=is_training) as arg_sc:
                return arg_sc


def rsoftmax(net): return tf.nn.softmax(net, axis=-1)
def rmul(gap, net): return gap * net


def rconv(net, channel, kernel): return slim.conv2d(net, channel, [
    kernel, kernel], stride=[1, 1], padding="same", activation_fn=None, normalizer_fn=None)


def split_conv(net, filters, i, j, r=2):
    with tf.variable_scope("resnest"+str(i)+"_"+str(j)):
        
        in_channel = net.get_shape()[-1]
        
        rgroups = tf.split(net, r, -1)
        rgroups = [rconv(g, filters, 3) for g in rgroups]  
        net = tf.concat(rgroups, axis=-1)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        rgroups = tf.split(net, r, axis=-1)
        gap = sum(rgroups)
        gap = tf.reduce_mean(gap, axis=[1, 2], keep_dims=True)  
        gap = slim.conv2d(gap, in_channel*r//4,
                          [1, 1], stride=[1, 1], padding="same")  
        gap = slim.conv2d(gap, filters * r, [1, 1], stride=[1, 1], padding="same",
                          activation_fn=None, normalizer_fn=None, biases_initializer=None)
        
        r_gap = tf.split(gap, r, -1)
        r_gap = [rsoftmax(g) for g in r_gap]  
        out = sum([rmul(r_gap[i], rgroups[i])
                   for i in range(len(rgroups))])  
        return out


def residual_block(net, in_ch, out_ch, rate, stride, i, j):
    shortcut = net
    net = slim.conv2d(net, in_ch, [1, 1], stride=[1, 1], padding="same")
    if stride:
        denet = slim.conv2d(net, in_ch, [3, 3], stride=[
                            2, 2], rate=rate, padding="same")
        net = slim.avg_pool2d(net, 3, stride=[2, 2], padding="same")
        net = split_conv(net, in_ch, i, j, r=2)
        shortcut = slim.conv2d(shortcut, out_ch, [1, 1], stride=[
                               2, 2], padding="same")
    else:
        denet = slim.conv2d(net, in_ch, [3, 3], stride=[
                            1, 1], rate=rate, padding="same")
        net = split_conv(net, in_ch, i, j, r=2)
        shortcut = slim.conv2d(shortcut, out_ch, [1, 1], stride=[
                               1, 1], padding="same")
    net = slim.conv2d(net, out_ch, [1, 1], stride=[1, 1], padding="same")
    return shortcut + net


def resnet_50_encoder(net, is_training):
    
    net = slim.conv2d(net, 64, [3, 3], stride=[2, 2],
                      padding="same", scope="conv1_1")
    net = slim.conv2d(net, 64, [3, 3], stride=[1, 1],
                      padding="same", scope="conv1_2")
    net = slim.conv2d(net, 128, [3, 3], stride=[
                      1, 1], padding="same", scope="conv1_3")
    net = slim.max_pool2d(net, [3, 3], stride=2, padding="same", scope='pool1')
    print("--------init---------", net)
    
    for i in range(3):
        net = residual_block(net, 64, 256, 1, False, 1, i)
    print("--------stage1---------", net)
    
    net = residual_block(net, 128, 512, 1, True, 1, 4)
    for i in range(3):
        net = residual_block(net, 128, 512, 1, False, 2, i)
    print("--------stage2---------", net)
    
    net = residual_block(net, 256, 1024, 1, False, 2, 4)
    for i in range(5):
        net = residual_block(net, 256, 1024, 2, False, 3, i)
    print("--------stage3---------", net)
    
    net = residual_block(net, 512, 2048, 2, False, 3, 5)
    for i in range(2):
        net = residual_block(net, 512, 2048, 4, False, 4, i)
    print("--------stage4---------", net)
    return net


def aspp(net, is_training):
    net1 = slim.conv2d(net, 256, [1, 1], stride=[1, 1], padding="same")
    net2 = slim.conv2d(net, 256, [3, 3], stride=[
                       1, 1], rate=12, padding="same")
    net3 = slim.conv2d(net, 256, [3, 3], stride=[
                       1, 1], rate=24, padding="same")
    net4 = slim.conv2d(net, 256, [3, 3], stride=[
                       1, 1], rate=36, padding="same")
    # different with source paper 28*28-> 7*7
    net5 = slim.avg_pool2d(net, 7, stride=[1, 1], padding="same")
    net5 = slim.conv2d(net5, 256, [1, 1], stride=[1, 1], padding="same")
    net = tf.concat([net1, net2, net3, net4, net5], axis=-1)
    net = slim.conv2d(net, 256, [1, 1], stride=[1, 1], padding="same")
    print("--------aspp---------", net)
    return net


def final(net, is_training, img_size):
    net = slim.conv2d(net, 256, [3, 3], stride=[1, 1], padding="same")
    net = slim.conv2d(net, 19, [1, 1], stride=[
                      1, 1], padding="same", activation_fn=None, normalizer_fn=None)
    if is_training:
        net = tf.image.resize_images(net, img_size)
    else:
        net = tf.image.resize_images(net, img_size)
    return net


def deeplabv3_resnest50(inputx, is_training, img_size):
    with slim.arg_scope(resnet_arg_scope(is_training)):
        net = resnet_50_encoder(inputx, is_training)
        net = aspp(net, is_training)
        out = final(net, is_training, img_size)
        print("--------out---------", out)
        return out
