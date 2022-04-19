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

import math
import tensorflow as tf
from src.layers import Conv2D, Conv3D, ActLayer, NormLayer
from src.utils.utils import to_pair

from .conv_module import Conv2DNormAct, Conv3DNormAct


class ResBlock(object):
    """A ResBlock class, consists of several conv blocks with bn.

    Args:
        num_blocks: int, number of conv blocks.
        mid_channels: int, number of the channels in the conv layers.
        res_scale: float, a scalar that scale the residual.
        act_cfg: dict, specify the activation `type` and other parameters.
        norm_cfg: dict, specify the normalization `type` and other parameters.
        use_spectral_norm: boolean, whether to use specatral normalization.
            Default False.
        trainable: boolean, whether in training phase. If True, the buffers will
            be add to UPDATE_OPS and update.
        name: str, variable scope name.
    """
    def __init__(self, num_blocks, mid_channels, res_scale=1.0,
                 act_cfg=dict(type='ReLU'), 
                 norm_cfg=dict(type='bn'), 
                 use_spectral_norm=False, 
                 trainable=True, name='ResBlock'):
        self.num_blocks = num_blocks
        self.output_channel = mid_channels
        self.res_scale = res_scale
        self.name = name
        self.trainable = trainable
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.use_spectral_norm = use_spectral_norm

    def shortcut_func(self, x):
        """Shortcut path. 
        
        May use a conv layer to change the number of channels.
        """
        c_in = x.get_shape().as_list()
        if c_in[-1] == self.output_channel:
            return x
        else:
            return Conv2D(self.output_channel, 
                          scale=self.scale,
                          name='conv_shortcut', 
                          use_spectral_norm=self.use_spectral_norm)(x)

    def build_block(self, x, index):
        """Build a basic conv block.
        """
        identity = self.shortcut_func(x)

        out = Conv2DNormAct(self.output_channel, scale=scales[0],
                            act_cfg=self.act_cfg, norm_cfg=self.norm_cfg,
                            use_spectral_norm=self.use_spectral_norm, 
                            name='conv{}a'.format(idx))(x)
        out = Conv2DNormAct(self.output_channel, scale=scales[1],
                            norm_cfg=self.norm_cfg, 
                            use_spectral_norm=self.use_spectral_norm, 
                            name='conv{}b'.format(idx))(out)

        return identity + out * self.res_scale

    def __call__(self, x):
        with tf.variable_scope(self.name) as scope:
            for i in range(self.num_blocks):
                x = self.build_block(x, i + 1)
            return x


class ResBlockNoBN(object):
    """A ResBlock class, consists of several conv blocks without bn.

    Args:
        num_blocks: int, number of conv blocks.
        mid_channels: int, number of the channels in the conv layers.
        res_scale: float, a scalar that scale the residual.
        act_cfg: dict, specify the activation `type` and other parameters.
        norm_cfg: dict, specify the normalization `type` and other parameters.
        use_spectral_norm: boolean, whether to use specatral normalization.
            Default False.
        trainable: boolean, whether in training phase. If True, the buffers will
            be add to UPDATE_OPS and update.
        name: str, variable scope name.
    """
    def __init__(self, num_blocks, mid_channels, res_scale=1.0, 
                 act_cfg=dict(type='ReLU'), dilation=1,
                 use_spectral_norm=False, trainable=True, 
                 name='ResBlockNoBN'):
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.res_scale = res_scale
        self.name = name
        self.trainable = trainable
        self.act_cfg = act_cfg
        self.dilation = (dilation, dilation)
        self.use_spectral_norm = use_spectral_norm

    def shortcut_func(self, x):
        """Shortcut path. May use a conv layer to change the number of channels.
        """
        c_in = x.get_shape().as_list()
        if c_in[-1] == self.output_channel:
            return x
        else:
            return Conv2D(self.output_channel, 
                          scale=self.scale,
                          name='conv_shortcut', 
                          use_spectral_norm=self.use_spectral_norm)(x)

    def build_block(self, x, idx):
        """Build a basic conv block.
        """
        out = Conv2D(self.mid_channels,
                     use_spectral_norm=self.use_spectral_norm,
                     name='conv{}a'.format(idx))(x)
        out = ActLayer(self.act_cfg)(out)
        out = Conv2D(self.mid_channels,
                     use_spectral_norm=self.use_spectral_norm, 
                     name='conv{}b'.format(idx))(out)
        return x + out * self.res_scale

    def __call__(self, x):
        with tf.variable_scope(self.name) as scope:
            for i in range(self.num_blocks):
                x = self.build_block(x, i + 1)
            return x
