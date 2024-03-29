#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG-16 for ImageNet.
Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper “Very Deep Convolutional Networks for
Large-Scale Image Recognition”  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.
Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow
Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping, as shown in the

The input image type is
    from scipy.misc import imread, imresize
    img1 = imread('data/laska.png', mode='RGB')  # test data in github
    img1 = imresize(img1, (224, 224))
So the input image is three channels, and is RGB.

following snippet:
#   >>> image_h, image_w, _ = np.shape(img)
#   >>> shorter_side = min(image_h, image_w)
#   >>> scale = 224. / shorter_side
#   >>> image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
#   >>> img = imresize(img, (image_h, image_w))
#   >>> crop_x = (image_w - 224) / 2
#   >>> crop_y = (image_h - 224) / 2
#   >>> img = img[crop_y:crop_y+224,crop_x:crop_x+224,:]
"""
from npu_bridge.npu_init import *

import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize
from nets.imagenet_classes import *
import os


def _conv_layers(net_in):
    with tf.name_scope('preprocess'):
        # Notice that we include a preprocessing layer that takes the RGB image
        # with pixels values in the range of 0-255 and subtracts the mean image
        # values (calculated over the entire ImageNet training set).
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean

    # conv1
    network = Conv2dLayer(
        net_in,
        act=tf.nn.relu,
        shape=[3, 3, 3, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 64, 64],  # 64 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool1')

    # conv2
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 64, 128],  # 128 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv2_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 128],  # 128 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool2')

    # conv3
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 128, 256],  # 256 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_2')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv3_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool3')

    # conv4
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 256, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_2')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv4_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool4')

    # conv5
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_1')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_2')
    network = Conv2dLayer(
        network,
        act=tf.nn.relu,
        shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='conv5_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool5')
    return network


def _fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
    return network


def get_vgg16(x, sess=None, pretrained=True):
    net_in = InputLayer(x, name='input')
    net_cnn = _conv_layers(net_in)  # simplified CNN APIs
    network = _fc_layers(net_cnn)

    if pretrained:
        npz = np.load('../model_weights/vgg16_weights.npz')
        params = []
        for val in sorted(npz.items()):
            print("  Loading %s" % str(val[1].shape))
            params.append(val[1])
        tl.files.assign_params(sess, params, network)
        return network
    else:
        tl.layers.initialize_global_variables(sess)
        return network


if __name__ == '__main__':
    DATA_PATH = '/home/aurora/workspaces2/PycharmProjects/tensorflow/tensorlayer/example/data'

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=npu_config_proto(config_proto=tfconfig)) as sess:
        network = get_vgg16(x, sess, pretrained=True)
        y = network.outputs
        network.print_params()
        network.print_layers()
        img1 = imread(os.path.join(DATA_PATH, 'laska.png'), mode='RGB')  # test data in github
        img1 = imresize(img1, (224, 224))
        probs = tf.nn.softmax(y)
        start_time = time.time()
        prob = sess.run(probs, feed_dict={x: [img1]})[0]
        print("  End time : %.5ss" % (time.time() - start_time))
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])
