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
import numpy as np

from .base_layer import BaseLayer

from src.runner.initializer import get_initializer, calculate_fan
from src.utils.utils import to_pair
from src.ops.weight_regularzation import spectral_norm
from src.ops import get_tensor_shape

__all__ = ["Conv2D", "Conv3D", "Conv2DTranspose", "Conv3DTranspose"]

class _ConvBaseLayer(BaseLayer):
    """A base class of convolution layer.

    y = conv(x, weights) + bias

    Properties:
        kernel: tensor, conv kernel.
        bias: tensor, bias tensor.
    """
    def __init__(self, 
                 num_filters, 
                 kernel_size=3, 
                 strides=1, 
                 dilations=1, 
                 use_bias=True,  
                 use_spectral_norm=False,
                 padding='same',
                 padding_mode='CONSTANT',
                 input_channels=None,
                 name='_conv_base'):
        """
        Initialization function of convolution base class.

        Args:
            num_filters: int, number of filters.
            kernel_size: int or list[int], the kernel size.
            strides: int or list[int], the stride size.
            dilations: int or list[int], the kernel dilations.
            use_bias: boolean, whether to use bias. Default True.
            use_spectral_norm: boolean, whether to use specatral normalization.
                Default False.
            padding: str or list[int]. If is given list of padding size, the 
                padding will be 'valid'. One can also pass in str such as 
                ('same', 'valid').
            padding_mode: str, indicating how to pad, i.e., REFLECT or CONSTANT.
            name: str, variable scope name.
        """
        self.num_filters = num_filters
        self.kernel_size = to_pair(kernel_size, 2)
        self.strides = to_pair(strides, 2)
        self.dilation = to_pair(dilations, 2)

        self.name = name
        self.use_bias = use_bias
        self.use_spectral_norm = use_spectral_norm
        self.padding = padding
        self.padding_mode = padding_mode

        self.in_channels = input_channels

    def __call__(self, x):
        """
        Execute function of forward.

        Args:
            x: tensor, input feature.
        
        Returns:
            tensor, convolved feature.
        """

        # Get the data type of the input.
        self.dtype = x.dtype
        if self.in_channels is None:
            self.in_channels = get_tensor_shape(x, dim=-1)

        # Get the weight and bias initializers.
        self.kernel_initializer = self.get_kernel_init(x)
        self.bias_initializer = self.get_bias_init(x)

        # Apply forward.
        with tf.variable_scope(self.name):
            x = self.apply_padding(x)
            x = self.forward(x)

        return x

    def apply_padding(self, x):
        """
        Do padding_mode before convolution. In 'same' padding_mode, the padding_mode will be 
        conducted by convolution operator itself.

        Args:
            x: tensor, input feature map.

        Returns:
            tensor, padded feature map or the original one.
        """
        # padding_mode for conv2d
        if isinstance(self.padding, (list, tuple)):
            if len(self.padding_mode) != 2:
                raise ValueError('Invalid padding_mode')
            padding_h, padding_w = self.padding_mode
            padding_new = ((0,0), 
                           (padding_h, padding_h), 
                           (padding_w, padding_w),
                           (0,0))
            x = tf.pad(x, padding_new, mode=self.padding_mode.upper())
            self.padding = 'Valid'
        elif self.padding_mode.upper() == 'REFLECT':
            padding_h = (self.kernel_size[0]//2, self.kernel_size[0]//2)
            padding_w = (self.kernel_size[1]//2, self.kernel_size[1]//2)
            padding_new = ((0,0), padding_h, padding_w, (0,0))
            x = tf.pad(x, padding_new, mode=self.padding_mode.upper())
            self.padding = 'Valid'
        return x

    def get_kernel_init(self, x):
        """
        Get kernel initializer. This function is called after passing through
        the input feature. We use 'kaiming_uniform' initializer.

        Args:
            x: tensor, input feature map.
        
        Returns:
            tensorflow initializer.
        """
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), 
            self.in_channels, 
            self.num_filters, 
            self.kernel_size, 
            dtype=self.dtype)
        return kernel_initializer

    def get_bias_init(self, x):
        """
        Get bias initializer. This function is called after passing through
        the input feature.

        Args:
            x: tensor, input feature map.
        
        Returns:
            tensorflow initializer.
        """
        fan = calculate_fan(self.kernel_size, self.in_channels)
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(
            -bound, 
            bound, 
            dtype=self.dtype)
        return bias_initializer

    @property
    def kernel(self):
        w = tf.get_variable(
            "kernel", 
            shape=[*self.kernel_size, self.in_channels, self.num_filters],
            initializer=self.kernel_initializer, 
            regularizer=None, 
            dtype=self.dtype)
        if self.use_spectral_norm:
            w = spectral_norm(w)
        return w

    @property
    def bias(self):
        bias = tf.get_variable(
            "bias", 
            [self.num_filters], 
            initializer=self.bias_initializer, 
            dtype=self.dtype)
        return bias

class Conv2D(_ConvBaseLayer):
    """A convolution2D class.
    """
    def __init__(self, 
                 num_filters, 
                 kernel_size=(3, 3), 
                 strides=(1, 1),
                 dilations=(1, 1), 
                 use_bias=True, 
                 use_spectral_norm=False,
                 padding='same',
                 padding_mode='CONSTANT',
                 input_channels=None,
                 name='Conv2D'):
        super().__init__(num_filters, 
                         kernel_size, 
                         strides, 
                         dilations, 
                         use_bias, 
                         use_spectral_norm,
                         padding,
                         padding_mode,
                         input_channels,
                         name)

    def forward(self, x):
        """
        Forward computation of the convolution 2d.

        Args:
            x: tensor, input feature map.
        
        Returns:
            tensor
        """
        x = tf.nn.conv2d(
            input=x, 
            filter=self.kernel,
            strides=[1, *self.strides, 1], 
            padding=self.padding.upper())
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class Conv2DTranspose(Conv2D):
    """A convolution transpose 2D class.
    """
    def __init__(self, 
                 num_filters, 
                 kernel_size=(3, 3), 
                 strides=(1, 1),
                 dilations=(1, 1), 
                 use_bias=True, 
                 use_spectral_norm=False,
                 padding='same',
                 padding_mode='CONSTANT',
                 input_channels=None,
                 name='Conv2DTranspose'):
        super().__init__(num_filters, 
                         kernel_size, 
                         strides, 
                         dilations, 
                         use_bias, 
                         use_spectral_norm,
                         padding,
                         padding_mode,
                         input_channels,
                         name)

    @property
    def kernel(self):
        # The kernel shape is (H_ksize, W_ksize, out_channels, in_channels), 
        # different from Conv2D.
        w = tf.get_variable(
            "kernel", 
            shape=[*self.kernel_size, self.num_filters, self.in_channels],
            initializer=self.kernel_initializer, 
            regularizer=None, 
            dtype=self.dtype)
        if self.use_spectral_norm:
            w = spectral_norm(w)
        return w

    def forward(self, x):
        """Forward computation of the convolution transpose 2d.

        Args:
            x: tensor, input feature map.
        """
        n, h, w, c = get_tensor_shape(x)
        output_shape = [n, 
                        h * self.strides[0], 
                        w * self.strides[1], 
                        self.num_filters]
        x = tf.nn.conv2d_transpose(
            input=x, 
            filter=self.kernel, 
            output_shape=output_shape,
            strides=[1, *self.strides, 1], 
            padding=self.padding.upper())
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class Conv3D(_ConvBaseLayer):
    """A convolution 3D class.
    """
    def __init__(self, 
                 num_filters, 
                 kernel_size=(3, 3, 3), 
                 strides=(1, 1, 1), 
                 dilations=(1, 1, 1), 
                 use_bias=True, 
                 use_spectral_norm=False,
                 padding='same',
                 padding_mode='CONSTANT',
                 input_channels=None,
                 name='Conv3D'):
        super().__init__(num_filters, 
                         kernel_size, 
                         strides, 
                         dilations, 
                         use_bias, 
                         use_spectral_norm,
                         padding,
                         padding_mode,
                         input_channels,
                         name)
        self.kernel_size = to_pair(kernel_size, 3)
        self.strides = to_pair(strides, 3)
        self.dilation = to_pair(dilations, 3)
    
    def apply_padding(self, x):
        """Do padding_mode before convolution. 
        In 'same' padding_mode, the padding_mode will be  conducted by 
        convolution operator itself.

        Args:
            x: tensor, input feature map.

        Returns:
            tensor, padded feature map or the original one.
        """
        # padding_mode for conv3d
        if type(self.padding) in [list, tuple]:
            if len(self.padding) != 3:
                raise ValueError('Invalid padding_mode')
            padding_d, padding_h, padding_w = self.padding
            padding_new = ((0,0), 
                            (padding_d, padding_d), 
                            (padding_h, padding_h), 
                            (padding_w, padding_w), (0,0))
            self.padding = 'Valid'
            x = tf.pad(x, padding_new, mode=self.padding_mode.upper())
        elif self.padding_mode.lower() == 'reflect':
            padding_d = (self.kernel_size[0]//2, self.kernel_size[0]//2)
            padding_h = (self.kernel_size[1]//2, self.kernel_size[1]//2)
            padding_w = (self.kernel_size[2]//2, self.kernel_size[2]//2)
            padding_new = ((0,0), padding_d, padding_h, padding_w, (0,0))
            x = tf.pad(x, padding_new, self.padding_mode.upper())
            self.padding = 'Valid'
        return x

    def forward(self, x):
        """Forward computation of the convolution 3d.

        Args:
            x: tensor, input feature map.
        """
        x = tf.nn.conv3d(
            input=x, 
            filter=self.kernel,
            strides=[1, *self.strides, 1], 
            padding=self.padding.upper())
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class Conv3DTranspose(Conv3D):
    """A convolution transpose 3D class.
    """
    def __init__(self, 
                 num_filters, 
                 kernel_size=(3, 3, 3), 
                 strides=(1, 1, 1), 
                 dilations=(1, 1, 1), 
                 use_bias=True, 
                 use_spectral_norm=False,
                 padding='same',
                 padding_mode='CONSTANT',
                 input_channels=None,
                 name='Conv3DTranspose'):
        super().__init__(num_filters, 
                         kernel_size, 
                         strides, 
                         dilations, 
                         use_bias, 
                         use_spectral_norm,
                         padding,
                         padding_mode,
                         input_channels,
                         name)

    @property
    def kernel(self):
        # The kernel shape is (H_ksize, W_ksize, out_channels, in_channels), 
        # different from Conv3D.
        w = tf.get_variable(
            "kernel", 
            shape=[*self.kernel_size, self.num_filters, self.in_channels],
            initializer=self.kernel_initializer, 
            regularizer=None, 
            dtype=self.dtype)
        if self.use_spectral_norm:
            w = spectral_norm(w)
        return w

    def forward(self, x):
        """Forward computation of the convolution transpose 3d.

        Args:
            x: tensor, input feature map.
        """
        n, h, w, c = get_tensor_shape(x)
        output_shape = [n, 
                        h * self.strides[0], 
                        w * self.strides[1], 
                        self.num_filters]
        x = tf.nn.conv3d_transpose(
            input=x, 
            filter=self.kernel, 
            output_shape=output_shape,
            strides=[1, *self.strides, 1], 
            padding=self.padding.upper())
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x
