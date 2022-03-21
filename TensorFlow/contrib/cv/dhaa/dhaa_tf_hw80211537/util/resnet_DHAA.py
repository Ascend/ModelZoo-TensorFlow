"""
Construct ResNet, SE_ResNet
Author: AjianLiu
Date: 2019/6/5
"""
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from numpy import *
import numpy as np
import util.transformer as transformer

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)
weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

##################################################################################
# Sampling function
##################################################################################
def flatten(x):
    return tf.layers.flatten(x)
def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return gap
def avg_pooling(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')
def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################
def relu(x):
    return tf.nn.relu(x)
def sigmoid(x):
    return tf.nn.sigmoid(x)

##################################################################################
# Normalization function
##################################################################################
def batch_norm(x, is_training=True, scope='bn'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None,
    is_training=is_training, scope=scope)

##################################################################################
# Layer
##################################################################################
def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=False, scope='conv'):
    with tf.compat.v1.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)
        return x

def fully_conneted(x, units, use_bias=False, scope='fully_0'):
    with tf.compat.v1.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x,
            units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

##################################################################################
# Block
##################################################################################
def resblock(net_type, x_init, channels, is_training=True,use_bias=False,downsample=0,stride = 1,scope='res_block_brach'):
    with tf.compat.v1.variable_scope(scope):
        if downsample == 0:
            x_brach1 = conv(x_init, channels, kernel=1, stride=stride, use_bias=use_bias, scope='downsample/0')
            x_brach1 = batch_norm(x_brach1, is_training, scope='downsample/1')
        else:
            x_brach1 = x_init

        x_brach2a = conv(x_init, channels, kernel=3, stride=stride, use_bias=use_bias, scope='conv1')
        x_brach2a = batch_norm(x_brach2a, is_training, scope='bn1')
        x_brach2a = relu(x_brach2a)
        x_brach2b = conv(x_brach2a, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv2')
        x_brach2b = batch_norm(x_brach2b, is_training, scope='bn2')

        x = x_brach1 + x_brach2b
        x = relu(x)
        return x

def bottle_resblock(net_type,x_init,channels,is_training=True,use_bias=False,downsample=0,stride = 1, scope='res_block_brach'):
    with tf.compat.v1.variable_scope(scope):
        if downsample == 0:
            x_brach1 = conv(x_init, channels*4,kernel=1, stride=stride, use_bias=use_bias, scope='downsample/0')
            x_brach1 = batch_norm(x_brach1, is_training, scope='downsample/1')
        else:
            x_brach1 = x_init

        x_brach2a = conv(x_init, channels, kernel=1, stride=stride, use_bias=use_bias, scope='conv1')
        x_brach2a = batch_norm(x_brach2a, is_training, scope='bn1')
        x_brach2a = relu(x_brach2a)

        x_brach2b = conv(x_brach2a, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv2')
        x_brach2b = batch_norm(x_brach2b, is_training, scope='bn2')
        x_brach2b = relu(x_brach2b)

        x_brach2c = conv(x_brach2b, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv3')
        x_brach2c = batch_norm(x_brach2c, is_training, scope='bn3')

        x = x_brach1 + x_brach2c
        x = relu(x)

        return x

##################################################################################
# Num_residual
##################################################################################
class RseNet_G(object):
    def __init__(self, res_n, mode=None):
        """
        :param res_n: '18, 34, 50, 101, 152'
        :param mode: 'resnet', 'se-resnet', 'resneXt'
        """
        self.res_n = res_n
        self.mode = mode
        self.residual_list = self.get_residual_layer(self.res_n)
        self.residual_block = self.get_residual_block(self.res_n)
    def get_residual_layer(self, res_n):
        x = []
        if res_n == 18:
            x = [2, 2, 2, 2]
        if res_n == 34:
            x = [3, 4, 6, 3]
        if res_n == 50:
            x = [3, 4, 6, 3]
        if res_n == 101:
            x = [3, 4, 23, 3]
        if res_n == 152:
            x = [3, 8, 36, 3]
        return x
    def get_residual_block(self, res_n):
        if (res_n == 18) or (res_n == 34):
            residual_block = resblock
        else:
            residual_block = bottle_resblock
        return residual_block

def define_resnet(ch, residual_list, residual_block, net_type, input_x, label_dim, is_training, reuse):
    with tf.compat.v1.variable_scope('', reuse=reuse):
        ### conv1
        x = conv(input_x, channels=ch, kernel=7, stride=2, scope='conv1')
        x = batch_norm(x, is_training, scope='bn1')
        x = relu(x)
        x = max_pooling(x)
        ### resblock1
        for i in range(0, residual_list[0]):
            x = residual_block(net_type, x,
                channels=ch, is_training=is_training, downsample=i, stride=1, scope='layer1/' + str(i))

        ### resblock2
        for i in range(0, residual_list[1]):
            stride = 2 if i ==0 else 1
            x = residual_block(net_type, x,
                channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2/' + str(i))
        ### resblock3
        for i in range(0, residual_list[2]):
            stride = 2 if i == 0 else 1
            x = residual_block(net_type, x,
                channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3/' + str(i))
        ### resblock4
        for i in range(0, residual_list[3]):
            stride = 2 if i == 0 else 1
            x = residual_block(net_type, x,
                channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4/' + str(i))
        ### global_avg_pooling
        gap_x = global_avg_pooling(x)
        logits_1 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_1/')
        logits_2 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_2/')
        logits_3 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_3/')
        logits_4 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_4/')
        logits_5 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_5/')
        logits_6 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_6/')
        logits_7 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_7/')
        logits_8 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_8/')
        logits_9 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_9/')
        logits_10 = fully_conneted(gap_x, units=label_dim, use_bias=False, scope='logits_10/')
    return logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, logits_9, logits_10

def conv_gap(x, dim=128, is_training=True, scope='conv_gap'):
    x = conv(x, channels=dim, kernel=1, stride=1, use_bias=False, scope=scope + '/conv1')
    x = batch_norm(x, is_training=is_training, scope=scope+'/conv1/bn')
    x = relu(x)

    x = global_avg_pooling(x)
    x = fully_conneted(x, units=dim, use_bias=False, scope=scope + '/fc')
    x = batch_norm(x, is_training=is_training, scope=scope+'/fc/bn')
    x = relu(x)

    return x

def aligned_region_pooling(input_x, thetas):
    outs = []
    for i in range(6):
        aligned_x = transformer.spatial_transformer_network(input_x, theta=thetas[:,i], out_dims=(56, 56))
        # aligned_x = tf.compat.v1.image.resize_bilinear(input_x, size=(56, 56))
        outs.append(aligned_x)
    return outs

def define_resnet_dhaa(ch, residual_list, residual_block, net_type, input_x, thetas, label_dim, is_training, reuse):
    with tf.compat.v1.variable_scope('', reuse=reuse):
        ### conv1
        x = conv(input_x, channels=ch, kernel=7, stride=2, scope='conv1')
        x = batch_norm(x, is_training, scope='bn1')
        x = relu(x)
        x = max_pooling(x)

        ### resblock1
        for i in range(0, residual_list[0]):
            x = residual_block(net_type, x,
            channels=ch, is_training=is_training, downsample=i, stride=1, scope='layer1/' + str(i))

        # [g1_x, g2_x, g3_x, l1_x, l2_x, l3_x] = [x * 1 for _ in range(6)]
        [g1_x, g2_x, g3_x, l1_x, l2_x, l3_x] = aligned_region_pooling(x, thetas)

        ### resblock2
        for i in range(0, residual_list[1]):
            stride = 2 if i == 0 else 1
            g1_x = residual_block(net_type, g1_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_g1/' + str(i))
            g2_x = residual_block(net_type, g2_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_g2/' + str(i))
            g3_x = residual_block(net_type, g3_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_g3/' + str(i))
            l1_x = residual_block(net_type, l1_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_l1/' + str(i))
            l2_x = residual_block(net_type, l2_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_l2/' + str(i))
            l3_x = residual_block(net_type, l3_x, channels=ch * 2, is_training=is_training, downsample=i, stride=stride, scope='layer2_l3/' + str(i))

        h1_x = tf.concat([g1_x, l1_x], axis=3)
        h2_x = tf.concat([g2_x, l2_x], axis=3)
        h3_x = tf.concat([g3_x, l3_x], axis=3)

        ### resblock3
        for i in range(0, residual_list[2]):
            stride = 2 if i == 0 else 1
            g1_x = residual_block(net_type, g1_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_g1/' + str(i))
            g2_x = residual_block(net_type, g2_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_g2/' + str(i))
            g3_x = residual_block(net_type, g3_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_g3/' + str(i))

            l1_x = residual_block(net_type, l1_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_l1/' + str(i))
            l2_x = residual_block(net_type, l2_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_l2/' + str(i))
            l3_x = residual_block(net_type, l3_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_l3/' + str(i))

            h1_x = residual_block(net_type, h1_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_h1/' + str(i))
            h2_x = residual_block(net_type, h2_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_h2/' + str(i))
            h3_x = residual_block(net_type, h3_x, channels=ch * 4, is_training=is_training, downsample=i, stride=stride, scope='layer3_h3/' + str(i))

        ### resblock4
        for i in range(0, residual_list[3]):
            stride = 2 if i == 0 else 1
            g1_x = residual_block(net_type, g1_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_g1/' + str(i))
            g2_x = residual_block(net_type, g2_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_g2/' + str(i))
            g3_x = residual_block(net_type, g3_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_g3/' + str(i))

            l1_x = residual_block(net_type, l1_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_l1/' + str(i))
            l2_x = residual_block(net_type, l2_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_l2/' + str(i))
            l3_x = residual_block(net_type, l3_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_l3/' + str(i))

            h1_x = residual_block(net_type, h1_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_h1/' + str(i))
            h2_x = residual_block(net_type, h2_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_h2/' + str(i))
            h3_x = residual_block(net_type, h3_x, channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_h3/' + str(i))

        g1_x = conv_gap(g1_x, dim=128, is_training=True, scope='g1_reduce/')
        g2_x = conv_gap(g2_x, dim=128, is_training=True, scope='g2_reduce/')
        g3_x = conv_gap(g3_x, dim=128, is_training=True, scope='g3_reduce/')

        l1_x = conv_gap(l1_x, dim=128, is_training=True, scope='l1_reduce/')
        l2_x = conv_gap(l2_x, dim=128, is_training=True, scope='l2_reduce/')
        l3_x = conv_gap(l3_x, dim=128, is_training=True, scope='l3_reduce/')

        h1_x = conv_gap(h1_x, dim=128, is_training=True, scope='h1_reduce/')
        h2_x = conv_gap(h2_x, dim=128, is_training=True, scope='h2_reduce/')
        h3_x = conv_gap(h3_x, dim=128, is_training=True, scope='h3_reduce/')

        #---------- logits for each sub-branch------
        logits_g1 = fully_conneted(g1_x, units=label_dim, use_bias=False, scope='g1/logits' )
        logits_g2 = fully_conneted(g2_x, units=label_dim, use_bias=False, scope='g2/logits' )
        logits_g3 = fully_conneted(g3_x, units=label_dim, use_bias=False, scope='g3/logits' )

        logits_l1 = fully_conneted(l1_x, units=label_dim, use_bias=False, scope='l1/logits' )
        logits_l2 = fully_conneted(l2_x, units=label_dim, use_bias=False, scope='l2/logits' )
        logits_l3 = fully_conneted(l3_x, units=label_dim, use_bias=False, scope='l3/logits' )

        logits_h1 = fully_conneted(h1_x, units=label_dim, use_bias=False, scope='h1/logits')
        logits_h2 = fully_conneted(h2_x, units=label_dim, use_bias=False, scope='h2/logits')
        logits_h3 = fully_conneted(h3_x, units=label_dim, use_bias=False, scope='h3/logits')

        # def RNN(x, time_steps=3, num_hidden=128, name='lstm'):
        #     from tensorflow.contrib import rnn
        #     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, name=name)
        #     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        #     return sum(outputs)
        # lstm_gx = RNN([g1_x, g2_x, g3_x], time_steps=3, num_hidden=128, name='g_lstm' )
        # lstm_lx = RNN([l1_x, l2_x, l3_x], time_steps=3, num_hidden=128, name='l_lstm' )
        # lstm_hx = RNN([h1_x, h2_x, h3_x], time_steps=3, num_hidden=128, name='h_lstm'  )
        # fusion_x = tf.concat([lstm_gx, lstm_lx, lstm_hx], axis=1)
        # logits = fully_conneted(fusion_x, units=label_dim, use_bias=False, scope='logits')

        fusion_hx = tf.concat([h1_x, h2_x, h3_x], axis=1)
        logits_h = fully_conneted(fusion_hx, units=label_dim, use_bias=False, scope='logits_h')
        # fusion_lx = tf.concat([l1_x, l2_x, l3_x], axis=1)
        # logits_l = fully_conneted(fusion_lx, units=label_dim, use_bias=False, scope='logits_l')

    # return logits, logits_g1, logits_g2, logits_g3, logits_l1, logits_l2, logits_l3, logits_h1, logits_h2, logits_h3
    return logits_h, logits_h1, logits_h2, logits_h3
    ### @1: logits_g1, logits_g2, logits_g3, logits_l1, logits_h1
    ### @2: logits_g1, logits_g2, logits_g3, logits_l1, logits_l2

def construct_resnet(res_n, input_x, thetas, batch_size, labels, label_dim=35, is_training = True, reuse=False):
    ch = 64
    resnet_G = RseNet_G(res_n)
    residual_block = resnet_G.residual_block
    residual_list = resnet_G.residual_list
    print('******** Network Info **********')
    print('*Resnet: residual={}, residual_list={}'.format(residual_block, residual_list))
    logits_list = define_resnet_dhaa(ch, residual_list, residual_block, None, input_x, thetas, label_dim, is_training, reuse)
    # logits_list = define_resnet(ch, residual_list, residual_block, None, input_x, label_dim, is_training, reuse)

    # -------------------
    cls_loss = 0
    for i in range(len(logits_list)):
        cls_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[:,0], logits=logits_list[i]))
    #### Norm for the prelogits
    reg_loss = tf.compat.v1.losses.get_regularization_loss()
    total_loss = cls_loss + reg_loss
    #------------
    logits = logits_list[0]
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32))
    temp = tf.convert_to_tensor(np.reshape(np.arange(label_dim, dtype=np.float32), (label_dim, 1)))
    prob = tf.nn.softmax(logits)
    pre_labels = tf.matmul(prob, temp)
    MAE = tf.reduce_mean(tf.abs(pre_labels - tf.cast(labels, tf.float32)))

    return logits, pre_labels, total_loss, accuracy, MAE







