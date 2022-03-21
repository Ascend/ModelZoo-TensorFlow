"""
model
"""
# coding=utf-8
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
# import tensorflow.compat.v1 as tf
import numpy as np


# from npu_bridge.npu_init import *

# res layer


def creat_layer(x, shape_w, train, lastLayer):
    """
    creat basic layer
    input:
    x: input tensor
    shape_w: conv size
    train: bool
    lastlayer: True->add res connect
    return:
    layer
    """
    W = tf.get_variable('w', shape=shape_w,
                        initializer=tf.glorot_uniform_initializer())
    # W = tf.get_variable('w', shape=shape_w, initializer=tf.constant_initializer(0.0))
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-7)(W))
    tf.add_to_collection(
        'losses', ((tf.nn.l2_loss(W) * 2) ** 0.5) * 0.5 * 1e-7)
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn = tf.layers.batch_normalization(
        x, training=train, fused=True, name='BN')
    if not (lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn


def residual_block(input_tensor, train, nLay):
    """
    build res block
    input:
    input:tensor
    train:bool
    nLay:layer num
    output:
    res block
    """
    lastLayer = False
    nw = {}
    nw['c' + str(0)] = input_tensor
    shape_w = {key: (3, 3, 64, 64) for key in range(2, nLay)}
    shape_w[1] = (3, 3, 2, 64)
    shape_w[nLay] = (3, 3, 64, 2)

    for i in np.arange(1, nLay + 1):
        if i == nLay:
            lastLayer = True
        with tf.variable_scope('Layer' + str(i)):
            nw['c' + str(i)] = creat_layer(nw['c' + str(i - 1)],
                                           shape_w[i], train, lastLayer)

    with tf.name_scope('Residual'):
        shortcut = tf.identity(input_tensor)
        dw = shortcut + nw['c' + str(nLay)]
    return dw


# class Aclass:
#     """
#     This class is created to do the data-consistency (DC) step as described in paper.
#     """
#     def __init__(self,mask,lam):
#         with tf.name_scope('Ainit'):
#             s=tf.shape(mask)
#             self.nrow,self.ncol=s[0],s[1]
#             self.pixels=self.nrow*self.ncol
#             self.mask=mask
#             self.lam=lam
#     def myAtA(self,img):
#         with tf.name_scope('AtA'):
#             kspace=  tf.fft2d(img)
#             temp=kspace*self.mask
#             coilImgs =tf.ifft2d(temp)
#             Imgs=coilImgs+self.lam*img
#         return Imgs

# def myCG(A,rhs):
#     """
#     This is my implementation of CG algorithm in tensorflow that works on
#     complex data and runs on GPU. It takes the class object as input.
#     """
#     r2c=lambda x:tf.complex(x[...,0],x[...,1])
#     c2r = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)
#     rhs=r2c(rhs)
#     cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-10)
#     def body(i,rTr,x,r,p):
#         with tf.name_scope('cgBody'):
#             Ap=A.myAtA(p)
#             alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
#             alpha=tf.complex(alpha,0.)
#             x = x + alpha * p
#             r = r - alpha * Ap
#             rTrNew = tf.to_float( tf.reduce_sum(tf.conj(r)*r))
#             beta = rTrNew / rTr
#             beta=tf.complex(beta,0.)
#             p = r + beta * p
#         return i+1,rTrNew,x,r,p

#     x=tf.zeros_like(rhs)
#     i,r,p=0,rhs,rhs
#     rTr = tf.to_float( tf.reduce_sum(tf.conj(r)*r),)
#     loopVar=i,rTr,x,r,p
#     out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
#     return c2r(out)


# def dc(rhs,mask,lam1):
#     """
#     This function is called to create testing model. It apply CG on each image
#     in the batch.
#     """
#     lam2=tf.complex(lam1,0.)
#     def fn( tmp ):
#         m,r=tmp
#         Aobj=Aclass( m,lam2 )
#         y=myCG(Aobj,r)
#         return y
#     inp=(mask,rhs)
#     rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn' )
#     return rec

# dc layer
def dc(out, x, mask, lam1):
    """
    build dc layer
    input:
    out: tensor
    x: original tensor
    mask: sampling mask
    lam1:lambda
    renturn:
    dc layer
    """

    # out,x,mask,lam1=input
    def r2c(x):
        """
        r2c
        """
        return tf.complex(x[..., 0], x[..., 1])

    def c2r(x):
        """
        c2r
        """
        return tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)

    out = r2c(out)
    x = r2c(x)
    lam1 = tf.cast(lam1, tf.complex64)
    kspace = tf.signal.fft2d(out)
    k0 = tf.signal.fft2d(x)
    temp = kspace * (1 - (lam1 / (0.5 + lam1)) * mask) + \
           (lam1 / (0.5 + lam1)) * k0 * mask
    coilImgs = tf.signal.ifft2d(temp)
    return c2r(coilImgs)


# def dc(out,x,mask,lam1):
#     rec=tf.map_fn(dc_,(out,x,mask,lam1),dtype=tf.float32,name='mapFn' )
#     return rec

def getLambda():
    """
    create a shared variable called lambda.
    return: lambda
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=0.5)
    return lam


# build model


def makeModel(x, mask, train, nLayers, K):
    """
    This is the main function that creates the model.
    input:
    x: tensor
    mask: sampling mask
    train: bool
    nLayers: res block layer num
    K: dc layer num
    return:
    model
    """
    with tf.compat.forward_compatibility_horizon(2019, 5, 1):
        out = {}
        out['dc0'] = x
        mask = tf.cast(mask, dtype=tf.complex64)
        with tf.name_scope('myModel'):
            with tf.variable_scope('Wts', reuse=tf.AUTO_REUSE):
                for i in range(1, K + 1):
                    j = str(i)
                    # residual block
                    out['dw' +
                        j] = residual_block(out['dc' + str(i - 1)], train, nLayers)
                    lam1 = getLambda()
                    # rhs= x + lam1 * out['dw' + j]
                    # out['dc'+j]=dc(rhs,mask,lam1)
                    out['dc' + j] = dc(out['dw' + j], x, mask, lam1)
    return out
