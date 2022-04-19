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
from .base_layer import BaseLayer


__all__ = ['ActLayer']

class ActLayer(BaseLayer):
    """Activation layer class.

    Args:
        cfg: dict, should specify the activation `type` and other parameters.
        name: str, scope name.
    """
    def __init__(self, cfg, name=None):
        super(ActLayer, self).__init__()
        self.type = cfg.get('type').lower()
        if self.type == 'leakyrelu':
            self.alpha = cfg.get('alpha', 0.2)
        elif self.type == 'prelu':
            # see https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html 
            # for explanation
            self.channelwise = cfg.get('channelwise', True)
        self.name = name

    def forward(self, x):
        if self.type == 'relu':
            return tf.nn.relu(x, name=self.name)
        elif self.type == 'elu':
            return tf.nn.elu(x, name=self.name)
        elif self.type == 'prelu':
            ndim = len(x.get_shape().as_list())
            if self.channelwise:
                num_parameters = x.get_shape().as_list()[-1]
            else:
                num_parameters = 1

            a = tf.get_variable(
                name=self.name+'_prelu_a', 
                shape=(num_parameters, ),
                dtype=x.dtype, 
                trainable=True, 
                initializer=tf.constant_initializer(0.25))

            if self.channelwise:
                a = tf.reshape(a, shape=tuple([1]*(ndim-1) + [num_parameters]))
                neg_mask = tf.cast(tf.less(x, 0.), dtype=x.dtype)
                neg_x = a * x   # apply parameter `a` channel-wise
                return x * (1. - neg_mask) + neg_x * neg_mask
            else:
                return tf.nn.leaky_relu(x, alpha=a, name=self.name)
        elif self.type == 'tanh':
            return tf.nn.tanh(x, name=self.name)
        elif self.type == 'leakyrelu':
            return tf.nn.leaky_relu(x, alpha=self.alpha, name=self.name)
        elif self.type == 'softplus':
            return tf.nn.softplus(x, name=self.name)
        elif self.type == 'sigmoid':
            return tf.nn.sigmoid(x, name=self.name)
        else:
            raise NotImplementedError
