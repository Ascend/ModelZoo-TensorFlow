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
import numpy as np
import tensorflow as tf

from src.runner.common import name_space
from src.layers import Conv2D, Conv3D, NormLayer, ActLayer, Linear
from src.utils.klass import get_subclass_given_name


def get_gan(cfg):
    """Get GAN instance given the configuration.

    Args:
        cfg: yacs node, config for the GAN.
    
    Returns:
        GAN instance.
    """

    try:
        klass = get_subclass_given_name(BaseGAN, cfg.loss.adversarial.gan_type)
    except IndexError:
        logger.error(f'Cannot find GAN type {cfg.loss.adversarial.gan_type}.')
        raise ValueError()

    return klass(cfg.loss.adversarial.mid_channels, cfg.loss.adversarial.norm_type)


class BaseGAN:
    """Base GAN class.
    """
    def __init__(self, scope=name_space.DiscriminatorVarScope):
        self.scope = scope

    def __call__(self, input):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            return self.forward(input)

    def forward(self, input):
        raise NotImplementedError


class VanillaGAN(BaseGAN):
    """A vanilla discriminator for 4D feature map.

    Args:
        mid_channels: int, multiplier of the channels in the middle layers.
        norm_type: str, type of the normalization layer.
        scope: str, discriminator scope name.
    """
    def __init__(self, mid_channels=64, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.norm_type = norm_type
        self.mid_channels = mid_channels
        self.kernel_size = (3, 3)

    def conv_norm_act(self, inputs, output_channel, kernel_size, stride, norm_type, is_train, scope):
        """A conv-norm-activation sequence.
        """
        with tf.variable_scope(scope):
            net = Conv2D(output_channel, kernel_size, stride, 
                         use_bias=norm_type=='in', 
                         name='conv',
                         use_spectral_norm=norm_type=='sn')(inputs)
            net = NormLayer(norm_type, is_train=is_train)(net)
            net = ActLayer(dict(type='leakyrelu', alpha=0.2), name='lrelu')(net)
        return net

    def forward(self, input):
        """Forward pass through the discriminator.
        """
        # no batchnorm for the first layer, output size [in_h/2, in_w/2]
        net = Conv2D(self.mid_channels, 
                     kernel_size=self.kernel_size,
                     strides=(1, 1),  
                     name='conv_first')(input)
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)

        # The discriminator block part
        # block 1, output size [in_h/4, in_w/4]
        net = self.conv_norm_act(net, self.mid_channels,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_1')
        # block 2, output size [in_h/8, in_w/8]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_2')
        # block 3, output size [in_h/16, in_w/16]
        net = self.conv_norm_act(net, self.mid_channels*3,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_3')
        # block_4, output size [in_h/32, in_w/32]
        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_4')

        # The dense layer 1
        b, h, w, c = net.get_shape().as_list()
        net = tf.reshape(net, [b, h * w * c])
        net = Linear(256, name='linear1')(net)  # channel-wise dense layer
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)
        net = Linear(1, name='linear_final')(net)  # channel-wise dense layer
        return net


class VanillaGAN3D(BaseGAN):
    """A vanilla discriminator for 5D feature map.

    Args:
        mid_channels: int, multiplier of the channels in the middle layers.
        norm_type: str, type of the normalization layer.
        scope: str, discriminator scope name.
    """
    def __init__(self, mid_channels=32, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.norm_type = norm_type
        self.mid_channels = mid_channels
        self.kernel_size = (3, 5, 5)

    def conv_norm_act(self, inputs, output_channel, kernel_size, stride, norm_type, is_train, scope):
        """A conv-norm-activation sequence.
        """
        with tf.variable_scope(scope):
            net = Conv3D(output_channel, kernel_size, stride, use_bias=norm_type=='in', name='conv',
                         use_spectral_norm=norm_type=='sn')(inputs)
            net = NormLayer(norm_type, is_train=is_train)(net)
            net = ActLayer(dict(type='leakyrelu', alpha=0.2), name='lrelu')(net)
        return net

    def forward(self, input):
        """Forward pass through the discriminator.
        """

        # no batchnorm for the first layer, output size [in_h/2, in_w/2]
        net = Conv3D(self.mid_channels, kernel_size=self.kernel_size, strides=(1, 1, 1),
                     name='conv_first')(input)
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)

        # The discriminator block part
        # block 1, output size [in_h/4, in_w/4]
        net = self.conv_norm_act(net, self.mid_channels,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_1')
        # block 2, output size [in_h/8, in_w/8]
        net = self.conv_norm_act(net, self.mid_channels,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_2')
        # block 3, output size [in_h/16, in_w/16]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_3')
        # block_4, output size [in_h/32, in_w/32]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_4')
        # block_5, output size [in_h/64, in_w/64]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_5')

        # The dense layer 1
        b, t, h, w, c = net.get_shape().as_list()
        net = tf.reshape(net, [b, t * h * w * c])
        net = Linear(256, name='linear1')(net)  # channel-wise dense layer
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)
        net = Linear(1, name='linear_final')(net)  # channel-wise dense layer
        return net


class PatchGAN(BaseGAN):
    """A PatchGAN discriminator for 4D feature map.

    Args:
        mid_channels: int, multiplier of the channels in the middle layers.
        norm_type: str, type of the normalization layer.
        scope: str, discriminator scope name.
    """
    def __init__(self, mid_channels=64, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.norm_type = norm_type
        self.mid_channels = mid_channels
        self.kernel_size = (3, 3)

    def conv_norm_act(self, inputs, output_channel, kernel_size, stride, norm_type, is_train, scope):
        """A conv-norm-activation sequence.
        """
        with tf.variable_scope(scope):
            net = Conv2D(output_channel, kernel_size, stride, use_bias=norm_type=='in', name='conv',
                         use_spectral_norm=norm_type=='sn')(inputs)
            net = NormLayer(norm_type, is_train=is_train)(net)
            net = ActLayer(dict(type='leakyrelu', alpha=0.2), name='lrelu')(net)
        return net

    def forward(self, input):
        """Forward pass through the discriminator.
        """

        # no batchnorm for the first layer, output size [in_h/2, in_w/2]
        net = Conv2D(self.mid_channels, kernel_size=self.kernel_size,
                     strides=(1, 1),  name='conv_first')(input)
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)

        # The discriminator block part
        # block 1, output size [in_h/4, in_w/4]
        net = self.conv_norm_act(net, self.mid_channels,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_1')
        # block 2, output size [in_h/8, in_w/8]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_2')
        # block 3, output size [in_h/16, in_w/16]
        net = self.conv_norm_act(net, self.mid_channels*3,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_3')
        # block_4, output size [in_h/32, in_w/32]
        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (2, 2), self.norm_type,
                                 True, 'disblock_4')

        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (1, 1), self.norm_type,
                                 True, 'disblock_5')
        net = Conv2D(self.mid_channels, kernel_size=(3, 3), strides=(1, 1),  name='conv_last')(net)
        return net


class PatchGAN3D(BaseGAN):
    """A PatchGAN discriminator for 5D feature map.

    Args:
        mid_channels: int, multiplier of the channels in the middle layers.
        norm_type: str, type of the normalization layer.
        scope: str, discriminator scope name.
    """
    def __init__(self, mid_channels=64, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.norm_type = norm_type
        self.mid_channels = mid_channels
        self.kernel_size = (3, 5, 5)

    def conv_norm_act(self, inputs, output_channel, kernel_size, stride, norm_type, is_train, scope):
        """
        A conv-norm-activation sequence.
        """
        with tf.variable_scope(scope):
            net = Conv3D(output_channel, kernel_size, stride, use_bias=norm_type=='in', name='conv',
                         use_spectral_norm=norm_type=='sn')(inputs)
            net = NormLayer(norm_type, is_train=is_train)(net)
            net = ActLayer(dict(type='leakyrelu', alpha=0.2), name='lrelu')(net)
        return net

    def forward(self, input):
        """
        Forward pass through the discriminator.
        """
        # no batchnorm for the first layer, output size [in_h/2, in_w/2]
        net = Conv3D(self.mid_channels, kernel_size=self.kernel_size,
                     strides=(1, 1, 1),  name='conv_first')(input)
        net = ActLayer(dict(type='leakyrelu', alpha=0.2))(net)

        # The discriminator block part
        # block 1, output size [in_h/4, in_w/4]
        net = self.conv_norm_act(net, self.mid_channels*2,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_1')
        # block 2, output size [in_h/8, in_w/8]
        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_3')
        # block 3, output size [in_h/16, in_w/16]
        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_5')
        # block_4, output size [in_h/32, in_w/32]
        net = self.conv_norm_act(net, self.mid_channels*4,
                                 self.kernel_size, (1, 2, 2), self.norm_type,
                                 True, 'disblock_7')

        net = Conv3D(self.mid_channels, kernel_size=self.kernel_size,
                     strides=(1, 1, 1),  name='conv_last')(net)
        # net = tf.reduce_mean(net, axis=1, keepdims=True)
        return net


class BigGAN(BaseGAN):
    """A BigGAN discriminator for 4D feature map.

    Args:
        mid_channels: int, multiplier of the channels in the middle layers.
        norm_type: str, type of the normalization layer.
        scope: str, discriminator scope name.
    """
    def __init__(self, mid_channels=64, norm_type='none', scope=name_space.DiscriminatorVarScope):
        """
        Initialization function of the discriminator.

        
        """
        super().__init__(scope)
        self.ch = mid_channels
        self.sn = False
        self.layer_num = 4

    def hw_flatten(self, x):
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def down_sample(self, x):
        return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

    def init_down_resblock(self, x_init, channels, use_bias=True, sn=False, scope='resblock'):
        with tf.variable_scope(scope):
            with tf.variable_scope('res1'):
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x_init)
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)

            with tf.variable_scope('res2'):
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)
                x = self.down_sample(x)

            with tf.variable_scope('shortcut'):
                x_init = self.down_sample(x_init)
                x_init = Conv2D(channels, kernel_size=(1, 1), use_bias=use_bias, use_spectral_norm=sn)(x_init)

            return x + x_init

    def down_resblock(self, x_init, channels, to_down=True, use_bias=True, sn=False, scope='resblock'):
        with tf.variable_scope(scope):
            init_channel = x_init.shape.as_list()[-1]
            with tf.variable_scope('res1'):
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x_init)
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)

            with tf.variable_scope('res2'):
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)
                if to_down:
                    x = self.down_sample(x)

            if to_down or init_channel != channels:
                with tf.variable_scope('shortcut'):
                    x_init = Conv2D(channels, kernel_size=(1, 1), use_bias=use_bias, use_spectral_norm=sn)(x_init)
                    if to_down:
                        x_init = self.down_sample(x_init)

            return x + x_init

    def google_attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = Conv2D(channels // 8, kernel_size=(1, 1), use_spectral_norm=self.sn, name='f')(x)  # [bs, h, w, c']
            f = tf.layers.max_pooling2d(f, pool_size=2, strides=2, padding='SAME')
            g = Conv2D(channels // 8, kernel_size=(1, 1), use_spectral_norm=self.sn, name='g')(x)  # [bs, h, w, c']
            h = Conv2D(channels // 2, kernel_size=(1, 1), use_spectral_norm=self.sn, name='h')(x)  # [bs, h, w, c]
            h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='SAME')

            # N = h * w
            s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, self.hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = Conv2D(channels, kernel_size=(1, 1), use_spectral_norm=self.sn, name='c')(o)  # [bs, h, w, c]
            x = gamma * o + x

        return x

    def forward(self, x):
        """Forward pass through the discriminator.
        """
        ch = self.ch
        x = self.init_down_resblock(x, channels=ch, sn=self.sn, scope='resblock_0')
        x = self.down_resblock(x, channels=ch * 2, sn=self.sn, scope='resblock_1')

        x = self.google_attention(x, channels=ch * 2, scope='self_attention')

        ch = ch * 2
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                x = self.down_resblock(x, channels=ch, sn=self.sn, to_down=False, scope='resblock_' + str(i+2))
            else:
                x = self.down_resblock(x, channels=ch * 2, sn=self.sn, scope='resblock_' + str(i+2))
            ch = ch * 2

        x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
        x = tf.reduce_sum(x, axis=[1, 2])
        x = Linear(1, name='linear')(x)
        return x

class MSPatchGAN(BaseGAN):
    """
    A multi-scale PatchGAN discriminator for 4D feature map.
    """
    def __init__(self, nf=64, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.nf = nf

    def patchGAN(self, x, n_layers, d_layers):
        x = Conv2D(self.nf,  kernel_size=(4, 4), strides=(2, 2), name='conv_first')(x)
        x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)

        for n in range(1, n_layers):
            x = Conv2D(self.nf * min(2 ** n, 8), kernel_size=(4, 4), strides=(1, 1), name='conv' + str(n))(x)
            x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
            if n < d_layers:
                x = Conv2D(self.nf * min(2 ** n, 8), kernel_size=(4, 4), strides=(2, 2), name='conv' + str(n) + '_down')(x)
        x = Conv2D(1, kernel_size=(4, 4), strides=(1, 1), name='conv_last')(x)
        return x

    def forward(self, x):
        n, h, w, c = x.get_shape().as_list()
        x_big = tf.image.resize_bilinear(x, size=(int(2*h), int(2*w)), align_corners=False, half_pixel_centers=False)
        x_mid = x
        x_sml = tf.image.resize_bilinear(x, size=(int(0.5*h), int(0.5*w)), align_corners=False, half_pixel_centers=False)
        with tf.variable_scope('big', reuse=tf.AUTO_REUSE):
            out_big = self.patchGAN(x_big, n_layers=3, d_layers=3)
        with tf.variable_scope('mid', reuse=tf.AUTO_REUSE):
            out_mid = self.patchGAN(x_mid, n_layers=3, d_layers=2)
        with tf.variable_scope('sml', reuse=tf.AUTO_REUSE):
            out_sml = self.patchGAN(x_sml, n_layers=3, d_layers=1)
        x = tf.concat([out_big, out_mid, out_sml], axis=-1)
        return x


class MSPatchBigGAN(BaseGAN):
    """
    A multi-scale PatchBigGAN discriminator for 4D feature map.
    """
    def __init__(self, nf=16, norm_type='in', scope=name_space.DiscriminatorVarScope):
        super().__init__(scope)
        self.nf = nf
        self.sn = False

    def hw_flatten(self, x):
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def down_sample(self, x):
        return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

    def init_down_resblock(self, x_init, channels, use_bias=True, sn=False, scope='resblock'):
        with tf.variable_scope(scope):
            with tf.variable_scope('res1'):
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x_init)
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)

            with tf.variable_scope('res2'):
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)
                x = self.down_sample(x)

            with tf.variable_scope('shortcut'):
                x_init = self.down_sample(x_init)
                x_init = Conv2D(channels, kernel_size=(1, 1), use_bias=use_bias, use_spectral_norm=sn)(x_init)

            return x + x_init

    def down_resblock(self, x_init, channels, to_down=True, use_bias=True, sn=False, scope='resblock'):
        with tf.variable_scope(scope):
            init_channel = x_init.shape.as_list()[-1]
            with tf.variable_scope('res1'):
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x_init)
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)

            with tf.variable_scope('res2'):
                x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
                x = Conv2D(channels, kernel_size=(3, 3), use_bias=use_bias, use_spectral_norm=sn)(x)
                if to_down:
                    x = self.down_sample(x)

            if to_down or init_channel != channels:
                with tf.variable_scope('shortcut'):
                    x_init = Conv2D(channels, kernel_size=(1, 1), use_bias=use_bias, use_spectral_norm=sn)(x_init)
                    if to_down:
                        x_init = self.down_sample(x_init)

            return x + x_init

    def patchGAN(self, x, n_layers, d_layers):
        x = self.init_down_resblock(x, channels=self.nf, sn=self.sn, scope='resblock_0')
        x = self.down_resblock(x, channels=self.nf * 2, sn=self.sn, scope='resblock_1')
        for n in range(n_layers):
            if n < d_layers:
                x = self.down_resblock(x, channels=self.nf * min(2 ** n, 8), sn=self.sn, scope='resblock_' + str(n + 2))
            else:
                x = self.down_resblock(x, channels=self.nf * min(2 ** n, 8), sn=self.sn, to_down=False, scope='resblock_' + str(n + 2))
        x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
        x = Conv2D(self.nf * 8, kernel_size=(3, 3), strides=(1, 1), use_spectral_norm=self.sn, name='conv')(x)
        x = ActLayer(dict(type='leakyrelu', alpha=0.2))(x)
        x = Conv2D(1, strides=(1, 1), name='conv_last')(x)
        return x

    def forward(self, x):
        n, h, w, c = x.get_shape().as_list()
        x_big = tf.image.resize_bilinear(x, size=(int(2*h), int(2*w)), align_corners=False, half_pixel_centers=False)
        x_mid = x
        x_sml = tf.image.resize_bilinear(x, size=(int(0.5*h), int(0.5*w)), align_corners=False, half_pixel_centers=False)
        with tf.variable_scope('big', reuse=tf.AUTO_REUSE):
            out_big = self.patchGAN(x_big, n_layers=3, d_layers=3)
        with tf.variable_scope('mid', reuse=tf.AUTO_REUSE):
            out_mid = self.patchGAN(x_mid, n_layers=3, d_layers=2)
        with tf.variable_scope('sml', reuse=tf.AUTO_REUSE):
            out_sml = self.patchGAN(x_sml, n_layers=3, d_layers=1)
        x = tf.concat([out_big, out_mid, out_sml], axis=-1)
        return x
