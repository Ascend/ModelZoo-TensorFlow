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
import tensorflow as tf

from src.layers import Conv2D, Conv3D, ActLayer, NormLayer, Conv2DTranspose


__all__ = ['Conv2DNormAct', 'Conv3DNormAct', 'Conv2DTransposeNormAct']


class Conv2DNormAct:
    """A base module consists of conv2d followed by norm and activation.

    Both normalization and activation layers are optional.

    Args:
        num_filters: int, number of filters.
        kernel_size: int or list[int], the kernel size.
        strides: int or list[int], the stride size.
        dilations: int or list[int], the kernel dilations.
        padding: str or list[int]. If is given list of padding size, the 
            padding will be 'valid'. One can also pass in str such as 
            ('same', 'valid').
        padding_mode: str, indicating how to pad, i.e., REFLECT or CONSTANT.

        use_bias: boolean, whether to use bias. Default True.
        use_spectral_norm: boolean, whether to use specatral normalization.
            Default False.
        trainable: boolean, whether in training phase. If True, the buffers will
            be add to UPDATE_OPS and update.
        act_cfg: dict, specify the activation `type` and other parameters.
        norm_cfg: dict, specify the normalization `type` and other parameters.
        name: str, variable scope name.
    """
    def __init__(self, num_filters, kernel_size=(3, 3), strides=(1, 1), 
                 dilations=(1, 1), padding='same', padding_mode='CONSTANT',
                 use_bias=True, use_spectral_norm=False, trainable=True, 
                 act_cfg=None, norm_cfg=None,
                 name='Conv2DModule'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations

        self.padding = padding
        self.padding_mode = padding_mode
        
        self.use_bias = use_bias
        self.use_spectral_norm = use_spectral_norm
        self.trainable = trainable

        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name):
            use_bias = self.use_bias if self.norm_cfg is None else False
            x = Conv2D(
                self.num_filters, kernel_size=self.kernel_size, strides=self.strides,
                dilations=self.dilations, use_bias=self.use_bias,
                use_spectral_norm=self.use_spectral_norm, 
                padding=self.padding, padding_mode=self.padding_mode,
                name='Conv2D')(x)

            if self.norm_cfg is not None:
                x = NormLayer(self.norm_cfg, is_train=self.trainable, name='Norm')(x)

            if self.act_cfg is not None:
                x = ActLayer(self.act_cfg)(x)
        return x


class Conv3DNormAct:
    """A base module consists of conv3d followed by norm and activation.

    Both normalization and activation layers are optional.

    Args:
        num_filters: int, number of filters.
        kernel_size: int or list[int], the kernel size.
        strides: int or list[int], the stride size.
        dilations: int or list[int], the kernel dilations.
        padding: str or list[int]. If is given list of padding size, the 
            padding will be 'valid'. One can also pass in str such as 
            ('same', 'valid').
        padding_mode: str, indicating how to pad, i.e., REFLECT or CONSTANT.

        use_bias: boolean, whether to use bias. Default True.
        use_spectral_norm: boolean, whether to use specatral normalization.
            Default False.
        trainable: boolean, whether in training phase. If True, the buffers will
            be add to UPDATE_OPS and update.
        act_cfg: dict, specify the activation `type` and other parameters.
        norm_cfg: dict, specify the normalization `type` and other parameters.
        name: str, variable scope name.
    """
    def __init__(self, num_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), 
                 dilations=(1, 1, 1), padding='same', padding_mode='CONSTANT',
                 use_bias=True, use_spectral_norm=False, trainable=True, 
                 act_cfg=None, norm_cfg=None,
                 name='Conv3DModule'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations

        self.padding = padding
        self.padding_mode = padding_mode
        
        self.use_bias = use_bias
        self.use_spectral_norm = use_spectral_norm
        self.trainable = trainable

        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name):
            use_bias = self.use_bias if self.norm_cfg is None else False
            x = Conv3D(self.num_filters, kernel_size=self.kernel_size, 
                       strides=self.strides, dilations=self.dilations, 
                       padding=self.padding, padding_mode=self.padding_mode,
                       use_bias=self.use_bias, use_spectral_norm=self.use_spectral_nor,
                       name='Conv3D')(x)

            if self.norm_cfg is not None:
                x = NormLayer(self.norm_cfg, is_train=self.trainable, name='Norm')(x)

            if self.act_cfg is not None:
                x = ActLayer(self.act_cfg)(x)
            return x


class Conv2DTransposeNormAct:
    """A base module consists of conv2d transpose followed by norm and activation.

    Both normalization and activation layers are optional.

    Args:
        num_filters: int, number of filters.
        kernel_size: int or list[int], the kernel size.
        strides: int or list[int], the stride size.
        dilations: int or list[int], the kernel dilations.
        padding: str or list[int]. If is given list of padding size, the 
            padding will be 'valid'. One can also pass in str such as 
            ('same', 'valid').
        padding_mode: str, indicating how to pad, i.e., REFLECT or CONSTANT.

        use_bias: boolean, whether to use bias. Default True.
        use_spectral_norm: boolean, whether to use specatral normalization.
            Default False.
        trainable: boolean, whether in training phase. If True, the buffers will
            be add to UPDATE_OPS and update.
        act_cfg: dict, specify the activation `type` and other parameters.
        norm_cfg: dict, specify the normalization `type` and other parameters.
        name: str, variable scope name.
    """
    def __init__(self, num_filters, kernel_size=(3, 3), strides=(1, 1), 
                 dilations=(1, 1), padding='same', padding_mode='CONSTANT',
                 use_bias=True, use_spectral_norm=False, trainable=True, 
                 act_cfg=None, norm_cfg=None,
                 name='Conv2DTransposeModule'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations

        self.padding = padding
        self.padding_mode = padding_mode
        
        self.use_bias = use_bias
        self.use_spectral_norm = use_spectral_norm
        self.trainable = trainable

        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name):
            use_bias = self.use_bias if self.norm_cfg is None else False
            x = Conv2DTranspose(
                self.num_filters, kernel_size=self.kernel_size, strides=self.strides,
                dilations=self.dilations, use_bias=self.use_bias,
                use_spectral_norm=self.use_spectral_norm, 
                padding=self.padding, padding_mode=self.padding_mode, 
                name='Conv2DTranspose')(x)

            if self.norm_cfg is not None:
                x = NormLayer(self.norm_cfg, is_train=self.trainable, name='Norm')(x)

            if self.act_cfg is not None:
                x = ActLayer(self.act_cfg)(x)
        return x
