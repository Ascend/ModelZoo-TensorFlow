#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseNet model.
"""

import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Dense, BatchNormalization
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Concatenate
from keras.utils.conv_utils import normalize_data_format
from keras.regularizers import l2
import keras.backend as K
import npu_convert_dropout

def dense_block(x, num_conv, filters, data_format,
                bottleneck=None, dropout_p=None, weight_decay=1e-4):
    """
    Returns a dense block which consists of Batch Normalization, ReLU and
    convolutional layers in that order.

    Parameters
    ----------
    model : Keras model
        The model to be extended with a dense block.
    num_conv : int
        The number of convolutional layers in the block.
    filters : int
        The number of filters for each convolutional layer (except the
        bottleneck layers, if any), aka the growth rate.
    bottleneck : int, default None
        If not None, then bottleneck layers are used and this parameter gives
        the number of feature maps that these layers must produce.
    dropout_p : float, default None
        The `p` of the Dropout layers. If None, then no Dropout is used.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    weight_decay : float, default 1e-4
        The weight decay to be used.

    Returns
    -------
    A dense block.
    """
    if data_format == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = -1

    for i in range(num_conv):
        if bottleneck is not None:
            y = BatchNormalization(axis=concat_axis)(x)
            y = Activation('relu')(y)
            y = Conv2D(bottleneck, (1,1), padding='same', use_bias=False,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay))(y)
            if dropout_p is not None:
                y = Dropout(dropout_p)(y)
            y = BatchNormalization(axis=concat_axis)(y)
        else:
            y = BatchNormalization(axis=concat_axis)(x)

        y = Activation('relu')(y)
        y = Conv2D(filters, (3,3), padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(y)
        if dropout_p is not None:
            y = Dropout(dropout_p)(y)

        # Merge feature maps
        if data_format == 'channels_last':
            x = Concatenate(axis=-1)([x, y])
        else:
            x = Concatenate(axis=1)([x, y])

    return x

def DenseNet(growth_rate=None, blocks=None, first_num_channels=16,
             dropout_p=None, bottleneck=None, compression=0.5,
             num_classes=1000, first_conv_pool=False,
             input_shape=(224, 224, 3), data_format=None, weight_decay=1e-4):
    """
    Returns a Keras model for the DenseNet-BC architecture.

    Parameters
    ----------
    growth_rate : int, default None
        The growth rate of the DenseNet, aka the `k` variable.
    blocks : list of int, default None
        A list of integers. Every integer indicates the number of
        convolutional layers in the corresponding block.
    first_num_channels : int, default 16
        The number of channels to be output by the first convolutional layer
        (before entering the first dense block).
    dropout_p : float, default None
        The `p` of the Dropout layers. If None, then no Dropout is used.
    bottleneck : int, default None
        If not None, then bottleneck layers are used and this parameter gives
        the number of feature maps that these layers must produce.
    compression : float
        The degree of compression to be performed in transition layers, aka the
        `theta` variable. The transition layer will output
        floor(`theta` * `k`) channels, so if `compression` == 1.0, then no
        compression is performed and the number of feature maps across
        transition layers remains unchanged.
    num_classes : int, default 1000
        The number of classes to be used for the final classification.
    first_conv_pool : bool, default False
        If True, then a max pooling layer is added after the first convolution
        (and before the first dense block).
    input_shape : tuple of int, default (224, 224)
        The input shape of the model.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    weight_decay : float, default 1e-4
        The weight decay to be used.

    Returns
    -------
    The Keras model.
    """
    inp = Input(shape=input_shape)

    if data_format is None:
        data_format=normalize_data_format(data_format)

    if data_format == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = -1

#    x = BatchNormalization(axis=concat_axis)(inp)
#    x = Activation('relu')(x)
    if first_conv_pool:
        x = Conv2D(first_num_channels, (7,7), strides=(2,2), use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   kernel_initializer='he_normal')(inp)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    else:
        x = Conv2D(first_num_channels, (3,3), strides=(1,1), use_bias=False,
                   kernel_regularizer=l2(weight_decay),
                   kernel_initializer='he_normal')(inp)

    # Add the dense blocks with transition layers in between
    num_channels = first_num_channels
    for i in range(len(blocks) - 1):
        # Add a dense block
        x = dense_block(x, blocks[i], growth_rate,
                        data_format=data_format,
                        bottleneck=bottleneck,
                        dropout_p=dropout_p,
                        weight_decay=weight_decay)

        num_channels += (blocks[i] * growth_rate)
        # Add a transition layer
        x = BatchNormalization(axis=concat_axis)(x)
        x = Activation('relu')(x)
        x = Conv2D(int(compression * num_channels), (1,1),
                   padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        num_channels = int(compression * num_channels)
        if dropout_p is not None:
            x = Dropout(dropout_p)(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)

    # Add the last dense block
    x = dense_block(x, blocks[-1], growth_rate,
                    data_format=data_format,
                    bottleneck=bottleneck, dropout_p=dropout_p,
                    weight_decay=weight_decay)

    # Add the classification layers
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax',
              kernel_regularizer=l2(weight_decay))(x)

    # Define the full model
    model = Model(inputs=inp, outputs=x)

    # Return the model
    return model
