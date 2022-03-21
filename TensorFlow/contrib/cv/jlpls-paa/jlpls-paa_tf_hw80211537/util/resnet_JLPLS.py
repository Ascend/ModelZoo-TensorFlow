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
import tensorflow.contrib as tf_contrib
from numpy import *
import numpy as np
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
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
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
                            units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def conv_gap(x, dim=128, is_training=True, scope='conv_gap'):
    x = conv(x, channels=dim, kernel=1, stride=1, use_bias=False, scope=scope + '/conv1')
    x = batch_norm(x, is_training=is_training, scope=scope+'/conv1/bn')
    x = relu(x)

    x = global_avg_pooling(x)
    x = fully_conneted(x, units=dim, use_bias=False, scope=scope + '/fc')
    x = batch_norm(x, is_training=is_training, scope=scope+'/fc/bn')
    x = relu(x)

    return x

def resblock(net_type, x_init, channels, is_training=True,use_bias=False,downsample=0, stride = 1,scope='res_block_brach',ratio=16):
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

def bottle_resblock(net_type,x_init,channels,is_training=True,use_bias=False,downsample=0,stride = 1, scope='res_block_brach',ratio=16):
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
        self.residual_block = self.get_residual_block(self.res_n, self.mode)
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

    def get_residual_block(self, res_n, mode):
        if (res_n == 18) or (res_n == 34):
            residual_block = resblock
        else:
            residual_block = bottle_resblock
        return residual_block

def define_resnet_two_branch(ch, residual_list, residual_block, net_type, input_x, label_dim, is_training, reuse):
    with tf.compat.v1.variable_scope('', reuse=reuse):
        ### conv1
        x = conv(input_x, channels=ch, kernel=7, stride=2, scope='conv1')
        x = batch_norm(x, is_training, scope='bn1')
        x = relu(x)
        x = max_pooling(x)
        ### resblock1
        for i in range(0, residual_list[0]):
            x = residual_block(net_type, x, channels=ch, is_training=is_training, downsample=i, stride=1, scope='layer1/' + str(i))

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
        ### resblock4 branch1
        x_branch1 = x * 1
        x_branch2 = x * 1
        for i in range(0, residual_list[3]):
            stride = 2 if i == 0 else 1
            x_branch1 = residual_block(net_type, x_branch1,
                channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_branch1/' + str(i))

        for i in range(0, residual_list[3]):
            stride = 2 if i == 0 else 1
            x_branch2 = residual_block(net_type, x_branch2,
                channels=ch * 8, is_training=is_training, downsample=i, stride=stride, scope='layer4_branch2/' + str(i))

    return x_branch1, x_branch2

def generate_penalty_binary(label_num, input_txt):

    pos_count = np.zeros(label_num, dtype=np.float32)
    count = 0.0
    with open(input_txt) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            labels = [int(x) for x in contents[1:]]

            labels = np.array( labels, dtype=np.float32 )
            pos_count += labels

            count += 1

    # count = count * np.ones(4, dtype=np.float32)
    ratio = pos_count / float(count)
    penalty_1 = np.sqrt( 1/( 2 * ratio ) )
    penalty_0 = np.sqrt( 1/( 2 * (1 - ratio) ) )

    penalty = np.stack( [penalty_0, penalty_1], axis=1 )
    return penalty

def generate_loss_weights( label_dim, train_file, batch_size, labels ):
    penalty_array = generate_penalty_binary(label_dim,  train_file )
    penalty_array = tf.convert_to_tensor(penalty_array)

    labels = tf.cast(labels, tf.int32)
    labels_for_ind = tf.reshape(labels, [-1])
    index = tf.one_hot(labels_for_ind, 2)

    penalty_array_t = tf.tile(penalty_array, [batch_size, 1])
    penalty_batch = tf.reduce_sum(penalty_array_t * index, 1)
    penalty_batch = tf.reshape(penalty_batch, [-1, label_dim])

    return penalty_batch

def weighted_bce_loss(penalty_batch, logits, labels):

    # -----------bce_loss----------
    labels = tf.cast(labels, tf.float32)
    bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='bce_loss')
    bce_loss = tf.reduce_mean(penalty_batch * bce_loss)

    return bce_loss

def build_sa_module(net, batch_size, scope, saN, num_attrs, is_training=True):
    with tf.variable_scope(scope, 'sa_module') as sc:
        fc_feas = []
        for i in range(saN):
            with tf.variable_scope('sa_%d' % (i + 1) ):
                conv1 = conv( net, channels=256, kernel=1, stride=1, scope='conv1')
                conv1 = batch_norm(conv1, is_training, scope='conv1/bn')
                conv1 = relu(conv1)

                conv2 = conv( conv1, channels=1, kernel=1, stride=1, scope='conv2')
                # conv2 = batch_norm(conv2, is_training, scope='conv2/bn')
                # conv2 = relu(conv2)

                # mask_shape = conv2.get_shape()
                mask_shape = [batch_size, 7, 7, 1]
                mask = tf.reshape(conv2, (batch_size, mask_shape[1]*mask_shape[2]), name='reshape1')
                mask = tf.nn.softmax(mask, name='softmax')
                mask = tf.reshape(mask, (batch_size,mask_shape[1],mask_shape[2],mask_shape[3]), name='reshape2')

                att_feas = net * mask

                # Global average pooling.
                att_feas = tf.reduce_mean(att_feas, [1, 2], name='pool5', keep_dims=True)
                att_feas = att_feas * 49

                fc = conv(att_feas, channels=128, kernel=1, scope='fc1')
                fc = batch_norm(fc, is_training, scope='fc1/bn')
                fc = relu(fc)
                fc_feas.append( fc )

        concat = tf.concat( fc_feas, axis=3 )
        fc = conv(concat, channels=num_attrs, kernel=1, scope='logits')
        fc = tf.squeeze(fc, [1, 2], name='SpatialSqueeze')
        # fc = tf.sigmoid(fc, 'sigmoid')

        return fc

def build_la_module(net, batch_size, scope, laN, num_attrs, is_training=True):
    with tf.variable_scope(scope, 'la_module') as sc:

        fea_conv1 = conv(net, channels=256, kernel=1, stride=1, scope='fea_conv1')
        fea_conv1 = batch_norm(fea_conv1, is_training=is_training, scope='fea_conv1/bn')
        fea_conv1 = relu( fea_conv1 )

        fea_conv2 = conv(fea_conv1, channels=laN*num_attrs, kernel=1, stride=1, scope='fea_conv2')
        # fea_conv2 = batch_norm(fea_conv2, is_training=is_training, scope='fea_conv2/bn')
        # fea_conv2 = relu(fea_conv2)

        mask_conv1 = conv(net, channels=256, kernel=1, stride=1, scope='mask_conv1')
        mask_conv1 = batch_norm(mask_conv1, is_training=is_training, scope='mask_conv1/bn')
        mask_conv1 = relu(mask_conv1)

        mask_conv2 = conv(mask_conv1, channels=laN*num_attrs, stride=1, scope='mask_conv2')

        # mask_shape = mask_conv2.get_shape()
        mask_shape = [batch_size, 7, 7, laN*num_attrs]

        mask = tf.transpose( mask_conv2, [0,3,1,2]  )
        mask = tf.reshape(mask, (batch_size*mask_shape[3], mask_shape[1]*mask_shape[2]), name='reshape1')
        mask = tf.nn.softmax(mask, name='softmax')
        mask = tf.reshape(mask, (batch_size, mask_shape[3], mask_shape[1], mask_shape[2]), name='reshape2')
        mask = tf.transpose(mask, [0, 2, 3, 1])

        att_feas = fea_conv2 * mask

        split_feas = tf.split( att_feas, num_attrs, axis=3)
        cons_fc = []
        #constrained loss
        for i in range(num_attrs):
            with tf.variable_scope('cons_%d' % (i + 1)):

                cons_conv_i = conv(split_feas[i], channels=10, kernel=1, stride=1, scope='cons_conv1')
                cons_conv_i = batch_norm(cons_conv_i, is_training, scope='cons_conv1/bn')
                cons_conv_i = relu(cons_conv_i)

                cons_conv_i = tf.reduce_mean(cons_conv_i, [1, 2], name='cons_pool5', keep_dims=True)
                cons_conv_i = cons_conv_i * 49

                cons_fc_i = fully_conneted(cons_conv_i, units=1, use_bias=False, scope='cons_logits')
                # cons_fc_i = tf.squeeze(cons_fc_i, [1, 2], name='cons_SpatialSqueeze')
                # cons_fc_i = tf.sigmoid(cons_fc_i, 'cons_sigmoid')

                cons_fc.append( cons_fc_i )
        cons_fc = tf.concat( cons_fc, axis = 1 )

        #la loss
        conv1 = conv(att_feas, channels=512, kernel=1, stride=1, scope='conv1')
        conv1 = batch_norm(conv1, is_training, scope='conv1/bn')
        conv1 = relu(conv1)

        conv2 = conv(conv1, channels=512, kernel=1, stride=1, scope='conv2')
        conv2 = batch_norm(conv2, is_training, scope='conv2/bn')
        conv2 = relu(conv2)

        pool5 = tf.reduce_mean(conv2, [1, 2], name='pool5', keep_dims=True)

        fc1 = fully_conneted(pool5, units=512, use_bias=False, scope='fc1')
        fc1 = batch_norm(fc1, is_training, scope='fc1/bn')
        fc1 = relu(fc1)

        fc2 = fully_conneted(fc1, units=num_attrs, use_bias=False, scope='logits')
        # fc2 = tf.squeeze(fc2, [1, 2], name='SpatialSqueeze')
        # fc2 = tf.sigmoid(fc2, 'sigmoid')

        return fc2, cons_fc


def construct_resnet(res_n, input_x,  batch_size, labels, label_dim=35, is_training = True, reuse=False, train_file=None):
    ch = 64
    resnet_G = RseNet_G(res_n, None)
    residual_block = resnet_G.residual_block
    residual_list = resnet_G.residual_list
    print('******** Network Info **********')
    print('*Resnet: residual={}, residual_list={}'.format(residual_block, residual_list))

    x_branch1, x_branch2 = \
        define_resnet_two_branch(ch, residual_list, residual_block, None, input_x, label_dim, is_training, reuse)
    # sa_logits = \
    #     build_sa_module(x_branch1, batch_size, 'sa_module', saN=6, num_attrs=label_dim, is_training=True)
    # la_logits, cons_logits = \
    #     build_la_module(x_branch2, batch_size, 'la_module', laN=10, num_attrs=label_dim, is_training=True)

    h1_gap = conv_gap(x_branch1, dim=128, is_training=True, scope='h1_reduce/')
    logits_h1 = fully_conneted(h1_gap, units=label_dim, use_bias=False, scope='h_1/logits')
    h2_gap = conv_gap(x_branch2, dim=128, is_training=True, scope='h2_reduce/')
    logits_h2 = fully_conneted(h2_gap, units=label_dim, use_bias=False, scope='h_2/logits')

    penalty_batch = generate_loss_weights(label_dim, train_file, batch_size, labels)
    # bce_loss = weighted_bce_loss(penalty_batch, sa_logits, labels) + \
    #            weighted_bce_loss(penalty_batch, la_logits, labels) + 0.2 * \
    #            weighted_bce_loss(penalty_batch, cons_logits, labels)

    bce_loss = weighted_bce_loss(penalty_batch, logits_h1, labels)
    bce_loss += weighted_bce_loss(penalty_batch, logits_h2, labels)

    #------------cal logits & pre labels & accuracy----------
    # logits = (sa_logits + la_logits) / 2
    logits = (logits_h1 + logits_h2) / 2

    pre_labels = logits > 0
    correct_prediction = tf.cast(tf.equal(tf.cast(pre_labels, tf.int64), tf.cast(labels, tf.int64)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    #### Norm for the prelogits
    reg_loss = tf.compat.v1.losses.get_regularization_loss()
    total_loss = bce_loss + reg_loss

    return logits, pre_labels, total_loss, accuracy







