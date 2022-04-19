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
import tensorflow as tf

from src.layers import Conv2D, NormLayer, ActLayer


class SPADE(object):
    """SPatially-Adaptive (DE)normalization. 
    
    See https://arxiv.org/pdf/1903.07291.pdf. The forward pass is borrowed from 
    https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py.
    
    Given the input input map \mathbf{h}, and a reference feature map \mathbf{m},
    the output activation value is :
    ```math
    \mathbf{y} = \gamma(\mathbf{m}) * \frac{\mathbf{h} - \mu}{\sigma} + \beta(\mathbf{m})
    ```
    where \gamma and \beta are both functions of \mathbf{m}.

    Args:
        num_filters: int, number of filters of the output tensor.
        kernel_size: int or list[int], kernel size of the conv layers.
        num_hidden: int, number of filters in the middle layers which compute the
            gamma and beta for normalization.
        training: boolean, indicating whether in training phase or not.
        norm_type: str, the type of normalization.
    """
    def __init__(self, num_filters, kernel_size=(3,3), 
                 num_hidden = 128, training=False, norm_type='in', 
                 return_all=False, name='spade',  
                 ver='v2'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.training = training
        self.norm_type = norm_type

    
    def spade(x, ref_feat, norm_nc, kernel_size=(3,3), name='spade', 
              nhidden = 128, training=False, norm_type='in', vis=False, 
              ver='v2'):
        with tf.variable_scope(name):
            # Part 1. generate parameter-free normalized activations
            normalized = NormLayer(norm_type=self.norm_type, center=False, 
                                   scale=False, is_train=training)(x)
    
            # Part 2. produce scaling and bias conditioned on reference map
            shape_x = x.get_shape().as_list()
            shape_label = ref_feat.get_shape().as_list()
            ref_feat = tf.image.resize_images(ref_feat, (shape_x[1], shape_x[2]),
                                              method=tf.image.ResizeMethod.BILINEAR,
                                              align_corners=True)

            if ver == 'v1':
                actv = Conv2D(nhidden, kernel_size=kernel_size, padding='SAME', 
                              strides=(1, 1), use_bias=True, trainable=True, 
                              name='mlp_shared')(ref_feat)
                actv = tf.nn.relu(actv)
            else:
                x_trans = Conv2D(shape_label[-1], kernel_size=kernel_size,
                                 padding='SAME', strides=(1, 1), use_bias=True, 
                                 trainable=True, name='mlp_trans')(x)
                actv = tf.nn.relu(ref_feat * x_trans)
                actv = Conv2D(nhidden, kernel_size=kernel_size, padding='SAME', 
                            strides=(1, 1), use_bias=True, trainable=True, 
                            name='mlp_shared')(actv)
                actv = tf.nn.relu(actv)

            gamma = Conv2D(norm_nc, kernel_size=kernel_size, padding='SAME', 
                        strides=(1, 1), use_bias=True, trainable=True, 
                        name='mlp_gamma')(actv)
            beta = Conv2D(norm_nc, kernel_size=kernel_size, padding='SAME', 
                        strides=(1, 1), use_bias=True, trainable=True, 
                        name='mlp_beta')(actv)

            # apply scale and bias
            out = normalized * (1 + gamma) + beta
            if vis:
                return out, gamma, beta
            return out


class SPADEResBlock:
    """ResBlock based on SPatially-Adaptive (DE)normalization. 
    
    See https://arxiv.org/pdf/1903.07291.pdf. The forward pass is borrowed from 
    https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py.
    
    Given the input input map \mathbf{h}, and a reference feature map \mathbf{m},
    the output activation value is :
    ```math
    \mathbf{y} = \gamma(\mathbf{m}) * \frac{\mathbf{h} - \mu}{\sigma} + \beta(\mathbf{m})
    ```
    where \gamma and \beta are both functions of \mathbf{m}.

    Args:
        num_filters: int, number of filters of the output tensor.
        kernel_size: int or list[int], kernel size of the conv layers.
        nhidden: int, number of filters in the middle layers which compute the
            gamma and beta for normalization.
        training: boolean, indicating whether in training phase or not.
        norm_type: str, the type of normalization.
    """
    def __init__(self, fin, fout, trainable=True, spectral_norm=False, with_spade=True, name='spade_res_block'):
        self.learned_short = fin != fout
        self.fmiddle = min(fin, fout)
        self.trainable = trainable
        self.fin = fin
        self.fout = fout
        self.with_spade = with_spade
        self.scope = name

    def __call__(self, x, ref):
        # TODO: extend to[NTHWC] 5D input
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            feat = x
            if self.with_spade:
                feat = spade(feat, ref, self.fin, kernel_size=(3, 3), name='spade1', training=self.trainable)
            feat = Conv2D(self.fmiddle, kernel_size=(3, 3), strides=(1, 1),
                          padding='SAME', trainable=self.trainable, name='conv1')(feat)
            feat = ActLayer(dict(type='leakyrelu', alpha=0.2))(feat)

            if self.with_spade:
                feat = spade(feat, ref, self.fmiddle, kernel_size=(3, 3), name='spade2', training=self.trainable)
            feat = Conv2D(self.fout, kernel_size=(3, 3), strides=(1, 1),
                          padding='SAME', trainable=self.trainable, name='conv2')(feat)
            feat = ActLayer(dict(type='leakyrelu', alpha=0.2))(feat)

            short_cut = x
            if self.learned_short:
                if self.with_spade:
                    short_cut = spade(short_cut, ref, self.fin, kernel_size=(3, 3),
                                      name='spade_shortcut', training=self.trainable)
                short_cut = Conv2D(self.fout, kernel_size=(3, 3), strides=(1, 1),
                                   padding='SAME', trainable=self.trainable, name='conv3')(short_cut)

            return short_cut + feat
