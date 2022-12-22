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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from Dataset.dataset import DataLoader
from builders import frontend_builder
import numpy as np

colors = np.array([[64,128,64],
[192,0,128],
[0,128, 192],
[0, 128, 64],
[128, 0, 0],
[64, 0, 128],
[64, 0, 192],
[192, 128, 64],
[192, 192, 128],
[64, 64, 128],
[128, 0, 192],
[192, 0, 64],
[128, 128, 64],
[192, 0, 192],
[128, 64, 64],
[64, 192, 128],
[64, 64, 0],
[128, 64, 128],
[128, 128, 192],
[0, 0, 192],
[192, 128, 128],
[128, 128, 128],
[64, 128,192],
[0, 0, 64],
[0, 64, 64],
[192, 64, 128],
[128, 128, 0],
[192, 128, 192],
[64, 0, 64],
[192, 192, 0],
[0, 0, 0],
[64, 192, 0]], dtype=np.float32)


def Upsampling(inputs, scale):
    return tf.compat.v1.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])


def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net


def AttentionRefinementModule(inputs, n_filters):
    inputs = slim.conv2d(inputs, n_filters, [3, 3], activation_fn=None)
    inputs = tf.nn.relu(slim.batch_norm(inputs, fused=True))

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net


def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net


class BiseNet(object):
    def __init__(self, model_config, train_config, num_classes, mode):
        self.model_config = model_config
        self.train_config = train_config
        self.num_classes = num_classes
        self.mode = mode
        assert mode in ['train', 'validation', 'inference', 'test']
        if self.mode == 'train':
            self.data_config = self.train_config['train_data_config']
        elif self.mode == 'validation':
            self.data_config = self.train_config['validation_data_config']
        elif self.mode == 'test':
            self.data_config = self.train_config['test_data_config']

        self.images = None
        self.images_feed = None
        self.labels = None
        self.net = None
        self.sup1 = None
        self.sup2 = None
        self.init_fn = None
        self.loss = None
        self.total_loss = None
        self.response = None

    def build_inputs(self):
        """Input fetching and batching

        Outputs:
          self.images: image batch of shape [batch, hz, wz, 3]
          labels: image batch of shape [batch, hx, wx, num_classes]
        """
        if self.mode in ['train', 'validation', 'test']:
            # Put data loading and preprocessing in CPU is substantially faster
            # DataSet prepare
            with tf.device("/cpu:0"):
                dataset = DataLoader(self.data_config, self.train_config['DataSet'], self.train_config['class_dict'])
                self.images, labels = dataset.get_one_batch()
                self.labels = tf.one_hot(labels, self.num_classes)

        else:
            self.images_feed = tf.placeholder(shape=[None, None, None, 3],
                                    dtype=tf.uint8, name='images_input')

            self.images = tf.cast(self.images_feed, dtype=tf.float32)/255

    def is_training(self):
        """Returns true if the model is built for training mode"""
        return self.mode == 'train'

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name='global_step',
            trainable=False,
            collections=[tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build_bisenet(self, reuse=False):
        """
        Builds the BiSeNet model.

        Arguments:
          reuse: Reuse variable or not

        Returns:
          BiSeNet model
        """

        ### The spatial path
        ### The number of feature maps for each convolution is not specified in the paper
        ### It was chosen here to be equal to the number of feature maps of a classification
        ### model at each corresponding stage
        batch_norm_params = self.model_config['batch_norm_params']
        init_method = self.model_config['conv_config']['init_method']

        if init_method == 'kaiming_normal':
            initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
        else:
            initializer = slim.xavier_initializer()

        with tf.compat.v1.variable_scope('spatial_net', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    spatial_net = ConvBlock(self.images, n_filters=64, kernel_size=[7, 7], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=64, kernel_size=[3, 3], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=64, kernel_size=[3, 3], strides=2)
                    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[1, 1])

        frontend_config = self.model_config['frontend_config']
        ### Context path
        logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(self.images, frontend_config,
                                                                                      self.is_training(), reuse)

        ### Combining the paths
        with tf.compat.v1.variable_scope('combine_path', reuse=reuse):
            with slim.arg_scope([slim.conv2d], biases_initializer=None, weights_initializer=initializer):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training(), **batch_norm_params):
                    # tail part
                    size = tf.shape(end_points['pool5'])[1:3]
                    global_context = tf.reduce_mean(end_points['pool5'], [1, 2], keep_dims=True)
                    global_context = slim.conv2d(global_context, 128, 1, [1, 1], activation_fn=None)
                    global_context = tf.nn.relu(slim.batch_norm(global_context, fused=True))
                    global_context = tf.compat.v1.image.resize_bilinear(global_context, size=size)

                    net_5 = AttentionRefinementModule(end_points['pool5'], n_filters=128)
                    net_4 = AttentionRefinementModule(end_points['pool4'], n_filters=128)

                    net_5 = tf.add(net_5, global_context)
                    net_5 = Upsampling(net_5, scale=2)
                    net_5 = ConvBlock(net_5, n_filters=128, kernel_size=[3, 3])
                    net_4 = tf.add(net_4, net_5)
                    net_4 = Upsampling(net_4, scale=2)
                    net_4 = ConvBlock(net_4, n_filters=128, kernel_size=[3, 3])

                    context_net = net_4

                    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=256)
                    net_5 = ConvBlock(net_5, n_filters=128, kernel_size=[3, 3])
                    net_4 = ConvBlock(net_4, n_filters=128, kernel_size=[3, 3])
                    net = ConvBlock(net, n_filters=64, kernel_size=[3, 3])
                    
                    # Upsampling + dilation or only Upsampling
                    net = Upsampling(net, scale=2)
                    net = slim.conv2d(net, 64, [3, 3], rate=2, activation_fn=tf.nn.relu, biases_initializer=None,
                                      normalizer_fn=slim.batch_norm)

                    net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, scope='logits')
                    self.net = Upsampling(net, 4)

                    # net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None, scope='logits')
                    # self.net = Upsampling(net, scale=8)

                    if self.mode in ['train', 'validation', 'test']:
                        sup1 = slim.conv2d(net_5, self.num_classes, [1, 1], activation_fn=None, scope='supl1')
                        sup2 = slim.conv2d(net_4, self.num_classes, [1, 1], activation_fn=None, scope='supl2')
                        self.sup1 = Upsampling(sup1, scale=16)
                        self.sup2 = Upsampling(sup2, scale=8)
                        self.init_fn = init_fn

    def build_loss(self):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.net, labels=self.labels))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.sup1, labels=self.labels))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.sup2, labels=self.labels))
        loss = loss1+loss2+loss3
        tf.compat.v1.losses.add_loss(loss)

        self.loss = loss1
        self.total_loss = tf.compat.v1.losses.get_total_loss()

        # shape = tf.shape(self.labels)

        # Tensorboard inspection
        
        """
        tf.compat.v1.summary.image('image', self.images, family=self.mode, max_outputs=1)
        tf.compat.v1.summary.image('GT', tf.reshape(
            tf.matmul(tf.reshape(self.labels, [-1, 32]), colors), [-1, shape[1], shape[2], 3]),
                         family=self.mode, max_outputs=1)
        tf.compat.v1.summary.image('response', tf.reshape(tf.matmul(
            tf.reshape(tf.one_hot(tf.argmax(self.net, -1), self.num_classes), [-1, 32]), colors),
            [-1, shape[1], shape[2], 3]), family=self.mode, max_outputs=1)
        tf.compat.v1.summary.scalar('total_loss', self.total_loss, family=self.mode)
        tf.compat.v1.summary.scalar('loss', self.loss, family=self.mode)
        """

        self.accuracy = tf.compat.v1.metrics.accuracy(predictions = tf.argmax(self.net, -1),
                                                        labels= tf.argmax(self.labels, -1))
        self.mean_IOU = tf.compat.v1.metrics.mean_iou(predictions=tf.argmax(self.net, -1),
                                                        labels=tf.argmax(self.labels, -1),
                                                        num_classes=self.num_classes)

        """
        with tf.control_dependencies([accuracy_update, mean_IOU_update]):
            tf.compat.v1.summary.scalar('accuracy', accuracy, family=self.mode)
            tf.compat.v1.summary.scalar('mean_IOU', mean_IOU, family=self.mode)
        """

    def predict(self):
        self.response = self.net

    def build(self, reuse=False):
        """Creates all ops for training and evaluation"""
        with tf.name_scope(self.mode):
            self.build_inputs()
            self.build_bisenet(reuse=reuse)
            if self.mode in ['train', 'validation', 'test']:
                self.build_loss()
            else:
                self.predict()

            if self.is_training():
                self.setup_global_step()




