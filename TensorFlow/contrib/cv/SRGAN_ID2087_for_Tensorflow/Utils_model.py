#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from npu_bridge.npu_init import *
class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))
    
def get_optimizer():
    # adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = tf.train.AdamOptimizer(learning_rate=1E-4, beta1=0.9, beta2=0.999, epsilon=1e-08)

    # loss_scale_manager = FixedLossScaleManager(loss_scale=2**10)
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 16, incr_every_n_steps=1000,
                                                           decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    opt = NPULossScaleOptimizer(opt, loss_scale_manager)

    return opt

