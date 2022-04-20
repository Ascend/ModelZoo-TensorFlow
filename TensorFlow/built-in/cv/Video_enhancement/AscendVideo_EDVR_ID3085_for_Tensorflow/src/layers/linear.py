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

from .base_layer import BaseLayer

from src.ops.weight_regularzation import spectral_norm
from src.runner.initializer import get_initializer, calculate_fan


__all__ = ["Linear"]

class Linear(BaseLayer):
    """A linear layer.

    y = x*weigths + bias

    Args:
        num_filters: int, number of filters in linear layer.
        use_bias: boolean, whether to apply bias. Default True.
        name: str, layer scope name.
        use_spectral_norm: boolean, whether to use spectral normalization
            on the weights. Default False.
        trainable: boolean, whether weights and bias are trainable.

    Attributes:
        kernel: tensor, linear kernel.
        bias: tensor, bias tensor.
    """
    def __init__(self, num_filters, use_bias=True, 
                 name='Linear', use_spectral_norm=False, 
                 trainable=True):
        self.num_filters = num_filters
        self.use_bias = use_bias
        self.name = name
        self.use_spectral_norm = use_spectral_norm
        self.trainable = trainable

    def get_kernel_init(self, x):
        """Get kernel initializer. 
        This function is called after passing through the input feature. 
        We use 'kaiming_uniform' initializer.
        """
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), 
            self.in_channels, 
            self.num_filters, 
            (1, ), 
            dtype=self.dtype)
        return kernel_initializer

    def get_bias_init(self, x):
        """Get bias initializer. 
        This function is called after passing through the input feature.
        """
        fan = calculate_fan((1, ), self.in_channels)
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

    def __call__(self, x):
        """Execute function of forward.

        Args:
            x: tensor, input feature.
        
        Returns:
            tensor, feature.
        """

        # Get the data type of the input.
        self.dtype = x.dtype
        self.in_channels = x.get_shape().as_list()[-1]

        # Get the weight and bias initializers.
        self.kernel_initializer = self.get_kernel_init(x)
        self.bias_initializer = self.get_bias_init(x)

        # Apply forward.
        with tf.variable_scope(self.name):
            x = self.forward(x)

        return x

    def forward(self, x):
        """Forward computation of the linear layer.
        """
        x = tf.layers.dense(
            x,
            units=self.num_filters,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            trainable=self.trainable,
            name=self.name,
        )
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x
