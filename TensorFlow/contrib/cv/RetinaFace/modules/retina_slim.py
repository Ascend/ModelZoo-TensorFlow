# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


import modules.resnet_v1.resnet_v1 as resnet_v1
from modules.anchor import decode_tf, prior_box_tf
# import resnet_v1.resnet_v1 as resnet_v1
# from anchor import decode_tf, prior_box_tf

# from Define import *


def batch_norm(x, is_training, eps=1e-05, decay=0.9, affine=True, scope='snow_batch_norm'):
    name = scope
    N, H, W, C = x.shape.as_list()
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        moving_mean = tf.get_variable('mean', [C], initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable('variance', [C], initializer=tf.ones_initializer, trainable=False)
        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(tf.cast(is_training, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

def group_normalization(x, is_training, G=32, EPS=1e-5, momentum=0.99, scope='snow_group_norm'):
    with tf.variable_scope(scope):
        # 1. [N, H, W, C] -> [N, C, H, W]
        # x = tf.transpose(x, [0, 3, 1, 2])
        x_shape = x.shape.as_list()
        N, H, W, C = [i if i is not None else -1 for i in x_shape]
        assert C % G == 0, 'The Group Number of GN is not supported! The channel number is '+str(C)

        # 2. reshape (group normalization)
        G = min(G, C)
        # x = tf.reshape(x, [N, H, W, G, C // G])
        # [N, H, W, C] -> [G, N, H, W, C//G]
        x = tf.split(x, G, -1)
        # [G, N, H, W, C//G] -> [N, H, W, G, C//G]
        x = tf.transpose(x, [1,2,3,0,4])

        # 3. create gamma, bete, moving_mean, moving_variance
        beta = tf.get_variable('beta', [C],
                                initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.ones_initializer)
        moving_mean = tf.get_variable('mean', [N,1,1,G,1],
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', [N,1,1,G,1],
                                          initializer=tf.ones_initializer,
                                          trainable=False)
        # 4. get mean, variance
        # mean, var = tf.nn.moments(x, [H, W, C], keep_dims=True)
        # mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, momentum),
                                          assign_moving_average(moving_variance, variance, momentum)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(tf.cast(is_training, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))

        # 5. normalize
        # x = (x - mean) / tf.sqrt(variance + EPS)
        # 6. gamma * x + beta
        # x = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        def normalize(x, mean, variance, scale, offset, variance_epsilon, name=None):
            with ops.name_scope(name, "normalization", [x, mean, variance, scale, offset]):
                inv = math_ops.rsqrt(variance + variance_epsilon)
                x = (x-math_ops.cast(mean, x.dtype))*math_ops.cast(inv, x.dtype)
                # x -> [N, H, W, G, C//G]
                # x = tf.reshape(x, [N, H, W, C])
                # [N, H, W, G, C//G] -> [G, N, H, W, C//G]
                x = tf.transpose(x, [3,0,1,2,4])
                # [G, N, H, W, C//G] -> G*[1, N, H, W, C//G]
                x = tf.split(x, G, 0)
                # G*[1, N, H, W, C//G] -> [1, N, H, W, C]
                x = tf.concat(x, -1)
                # [1, N, H, W, C] -> [N, H, W, C]
                x = tf.squeeze(x, 0)
                if scale is not None:
                    x *= math_ops.cast(scale, x.dtype)
                # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
                # the precise order of ops that are generated by the expression below.
                return x + math_ops.cast(offset, x.dtype) if offset is not None else x
        x = normalize(x, mean, variance, gamma, beta, EPS)
    return x


def conv_gn_relu(x, wd, filters, kernel_size, strides, padding, is_training, scope, gn=True, activation=None, use_bias=True, upscaling=False):
    k_regularizer = tf.contrib.layers.l2_regularizer(wd)
    k_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
    # class_bias_initializer = tf.constant_initializer(value=-np.log((1 - 0.01) / 0.01))
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                 kernel_initializer=k_initializer, kernel_regularizer=k_regularizer, use_bias=use_bias, name='conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                           kernel_initializer=k_initializer, kernel_regularizer=k_regularizer, use_bias=use_bias, name='upconv2d')

        if gn:
            # x = group_normalization(x, is_training=is_training)
            x = batch_norm(x, is_training=is_training)

        if activation is not None:
            if activation == 'relu':
                x = tf.nn.relu(x, name='relu')
            elif activation == 'lrelu':
                x = tf.nn.leaky_relu(x, alpha=0.1, name='leaky_relu')
            else:
                raise NotImplementedError(
                    'Activation function type {} is not recognized.'.format(activation))
    return x


def connection_block(x1, x2, wd, out_ch, is_training, scope):
    assert out_ch % 4 == 0, 'Check the out put channel'
    act = 'relu'
    if (out_ch <= 64):
        act = 'lrelu'
    with tf.variable_scope(scope):
        up_h, up_w = tf.shape(x2)[1], tf.shape(x2)[2]
        x1 = tf.image.resize(x1, [up_h, up_w], method='nearest')
        # x1 = conv_gn_relu(x1, out_ch, (3, 3), 2, 'same', is_training, 'Upsampling', upscaling=True)
        x2 = conv_gn_relu(x2, wd, out_ch, [1, 1], 1,
                          'valid', is_training, 'Conv1x1_1', activation=act)
        output = x2+x1
        x = conv_gn_relu(output, wd, out_ch, [3, 3], 1,
                         'same', is_training, 'Conv3x3_1', activation=act)
    return x


def ResNet50_FPN(input_var, wd, out_ch, is_training, reuse=False):
    # OpenCV BGR to RGB & normalize (ImageNet)
    # x = input_var[..., ::-1] - MEAN
    x = input_var
    # RES50
    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(
            x, is_training=is_training, reuse=reuse)

    pyramid_dic = {}
    # old output
    # pyramid_dic['C2'] = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    # pyramid_dic['C3'] = end_points['resnet_v1_50/block1']
    # pyramid_dic['C4'] = end_points['resnet_v1_50/block2']
    # pyramid_dic['C5'] = end_points['resnet_v1_50/block4']
    '''
    Tensor("resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0", shape=(8, 160, 160, 256), dtype=float32)
    Tensor("resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0", shape=(8, 80, 80, 256), dtype=float32)
    Tensor("resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0", shape=(8, 40, 40, 512), dtype=float32)
    Tensor("resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0", shape=(8, 20, 20, 2048), dtype=float32)
    '''
    # as the same as the retinaface paper
    pyramid_dic['C2'] = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    pyramid_dic['C3'] = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']
    pyramid_dic['C4'] = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
    pyramid_dic['C5'] = end_points['resnet_v1_50/block4']
    '''
    Tensor("resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0", shape=(8, 160, 160, 256), dtype=float32)
    Tensor("resnet_v1_50/block2/unit_3/bottleneck_v1/Relu:0", shape=(8, 80, 80, 512), dtype=float32)
    Tensor("resnet_v1_50/block3/unit_5/bottleneck_v1/Relu:0", shape=(8, 40, 40, 1024), dtype=float32)
    Tensor("resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0", shape=(8, 20, 20, 2048), dtype=float32)
    '''
    # print(pyramid_dic['C2'])
    # print(pyramid_dic['C3'])
    # print(pyramid_dic['C4'])
    # print(pyramid_dic['C5'])

    # FPN
    '''
                C6/P6--->P6
                        ^
                        |
                    Conv3*3,2,Down
                        |
    C5--->Conv1*1,1--->P5
                        |
                    Conv3*3,2,Up
                        |
                        V
    C4--->Conv1*1,1------->Conv3*3,1--->P4
                                        |
                                    Conv3*3,2,Up
                                        |
                                        V
    C3--->Conv1*1,1----------------------->Conv3*3,1--->P3
                                                        |
                                                    Conv3*3,2,Up
                                                        |
                                                        V
    C2--->Conv1*1,1--------------------------------------->Conv3*3,1--->P2
    '''
    act = 'relu'
    if (out_ch <= 64):
        act = 'lrelu'
    with tf.variable_scope('FPN', reuse=reuse):
        # # build P6
        # x = pyramid_dic['C5']
        # with tf.variable_scope('P6_Conv'):
        #     # x = conv_gn_relu(x, 256, (1, 1), 1, 'valid', is_training, 'Conv1*1_1')
        #     x = conv_gn_relu(x, wd, out_ch, (3, 3), 2, 'same',
        #                      is_training, 'Conv3x3_2', activation=act)
        # pyramid_dic['P6'] = x

        # build P6, P5
        x = pyramid_dic['C5']
        with tf.variable_scope('P5_Conv'):
            x = conv_gn_relu(x, wd, out_ch, (1, 1), 1, 'valid',
                             is_training, 'Conv1x1_1', activation=act)
        with tf.variable_scope('P6_Conv'):
            # x = conv_gn_relu(x, 256, (1, 1), 1, 'valid', is_training, 'Conv1*1_1')
            y = conv_gn_relu(x, wd, out_ch, (3, 3), 2, 'same',
                             is_training, 'Conv3x3_2', activation=act)
        pyramid_dic['P6'] = y
        pyramid_dic['P5'] = x
        # build P4
        x = pyramid_dic['P5']
        x = connection_block(
            x, pyramid_dic['C4'], wd, out_ch, is_training, 'P4_conv')
        pyramid_dic['P4'] = x
        # build P3
        x = pyramid_dic['P4']
        x = connection_block(
            x, pyramid_dic['C3'], wd, out_ch, is_training, 'P3_conv')
        pyramid_dic['P3'] = x
        # build P2
        x = pyramid_dic['P3']
        x = connection_block(
            x, pyramid_dic['C2'], wd, out_ch, is_training, 'P2_conv')
        pyramid_dic['P2'] = x
        '''
        # P2 : Tensor("RetinaNet/P2_conv/relu:0", shape=(8, 160, 160, 256), dtype=float32)
        # P3 : Tensor("RetinaNet/P3_conv/relu:0", shape=(8, 80, 80, 256), dtype=float32)
        # P4 : Tensor("RetinaNet/P4_conv/relu:0", shape=(8, 40, 40, 256), dtype=float32)
        # P5 : Tensor("RetinaNet/P5_conv/relu:0", shape=(8, 20, 20, 256), dtype=float32)
        # P6 : Tensor("RetinaNet/P6_conv/relu:0", shape=(8, 10, 10, 256), dtype=float32)
        '''

    return pyramid_dic


def SSH(input_var, wd, out_ch, is_training, name='SSH', reuse=False):
    x = input_var
    assert out_ch % 4 == 0, 'Check the out put channel'
    act = 'relu'
    if (out_ch <= 64):
        act = 'lrelu'
    with tf.variable_scope(name, reuse=reuse):
        conv_3x3 = conv_gn_relu(x, wd, out_ch//2, (3, 3), 1, 'same',
                                is_training, 'Conv3x3', activation=None)
        y = conv_gn_relu(x, wd, out_ch//4, (3, 3), 1, 'same',
                         is_training, 'Conv5x5', activation=act)
        conv_5x5 = conv_gn_relu(y, wd, out_ch//4, (3, 3), 1, 'same',
                                is_training, 'Conv5x5_1', activation=None)
        z = conv_gn_relu(y, wd, out_ch//4, (3, 3), 1, 'same',
                         is_training, 'Conv7x7', activation=act)
        conv_7x7 = conv_gn_relu(z, wd, out_ch//4, (3, 3), 1, 'same',
                                is_training, 'Conv7x7_1', activation=None)
        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = tf.nn.relu(output, name='relu')
    return output


def BboxHead(input_var, num_anchor, name='BboxHead', reuse=False):
    x = input_var
    n, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    with tf.variable_scope(name, reuse=reuse):
        x = tf.layers.conv2d(inputs=x, filters=num_anchor *
                             4, kernel_size=1, strides=1)
        output = tf.reshape(x, [n, h * w * num_anchor, 4])
    return output


def LandmarkHead(input_var, num_anchor, name='LandmarkHead', reuse=False):
    x = input_var
    n, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    with tf.variable_scope(name, reuse=reuse):
        x = tf.layers.conv2d(inputs=x, filters=num_anchor *
                             10, kernel_size=1, strides=1)
        output = tf.reshape(x, [n, h * w * num_anchor, 10])
    return output


def ClassHead(input_var, num_anchor, name='ClassHead', reuse=False):
    x = input_var
    n, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    with tf.variable_scope(name, reuse=reuse):
        x = tf.layers.conv2d(inputs=x, filters=num_anchor *
                             2, kernel_size=1, strides=1)
        output = tf.reshape(x, [n, h * w * num_anchor, 2])
    return output


def RetinaFaceModel(cfg, training=False, name='RetinaFaceModel'):
    # root_path = cfg['root_path']
    # input_size = cfg['input_size'] if training else None
    iou_th = cfg['iou_threshold']
    score_th = cfg['score_threshold']
    num_anchor = len(cfg['min_sizes'][0])
    backbone_type = cfg['backbone_type']
    wd = cfg['weights_decay']
    out_ch = cfg['out_channel']

    def retina_face_r50(input_var, reuse=False):
        pyramid_dic = ResNet50_FPN(
            input_var, wd, out_ch, training, reuse=reuse)
        # feature oder: ['P2', 'P3', 'P4', 'P5', 'P6']
        FPN_Features = [pyramid_dic[i] for i in pyramid_dic.keys() if 'P' in i]
        FPN_Features = FPN_Features[::-1]
        SSH_Feature = [SSH(f, wd, out_ch, training, name=f'SSH_{i}', reuse=reuse)
                       for i, f in enumerate(FPN_Features)]
        Bbox_Regressions = tf.concat([BboxHead(f, num_anchor, name=f'BboxHead_{i}', reuse=reuse)
                                      for i, f in enumerate(SSH_Feature)], axis=1)
        Landm_Regressions = tf.concat([LandmarkHead(f, num_anchor, name=f'LandmarkHead_{i}', reuse=reuse)
                                       for i, f in enumerate(SSH_Feature)], axis=1)
        ClassHead_Regressions = tf.concat([ClassHead(f, num_anchor, name=f'ClassHead_{i}', reuse=reuse)
                                           for i, f in enumerate(SSH_Feature)], axis=1)
        classifications = tf.nn.softmax(ClassHead_Regressions, axis=-1)
        if training:
            out = (Bbox_Regressions, Landm_Regressions, classifications)
        else:
            # only for batch size 1
            preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
                [Bbox_Regressions[0], Landm_Regressions[0],
                 tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
                 classifications[0, :, 1][..., tf.newaxis]], 1)

            priors = prior_box_tf((tf.shape(input_var)[1], tf.shape(input_var)[2]),
                                  cfg['min_sizes'],  cfg['steps'], cfg['clip'])

            decode_preds = decode_tf(preds, priors, cfg['variances'])
            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=iou_th,
                score_threshold=score_th)

            out = tf.gather(decode_preds, selected_indices)

        return out

    model_func = retina_face_r50 if backbone_type == 'ResNet50' else None
    return model_func

def test_gn(training=False):
    def group_norm(x):
        # return batch_norm(x, is_training=training, name='snowball_bn')
        return group_normalization(x, is_training=training)
        # return GN(x)
    return group_norm

'''
if __name__ == '__main__':
    cfg = {'backbone_type': 'ResNet50', 'weights_decay': 5e-4, 'out_channel': 256, 'min_sizes': [[16, 32], [
        64, 128], [256, 512]], 'iou_threshold': 0.4, 'score_threshold': 0.02, 'steps': [8, 16, 32], 'clip': False, 'variances': [0.1, 0.2]}
    input_var = tf.placeholder(tf.float32,
                               [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    model = RetinaFaceModel(cfg, True)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    print(reg_losses)
    output = model(input_var)
'''
