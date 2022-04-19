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
import tensorflow.contrib.slim as slim

from .base_layer import BaseLayer


__all__ = ['NormLayer']

EPSILON = 1e-5
DECAY = 0.99

def batch_norm(x, center=True, scale=True, is_train=True):
    """Batch normalization function.
    
    y = (x - \mu) / \sigma * \gamma + \beta

    Args:
        x: tensor, input feature map.
        center: boolean, whether to use bias, i.e. beta.
        scale: bollean, whether to use affine parameters, i.e. \gamma.
        is_train: boolean, whether to update the buffers (moving average and var).

    Returns:
        tensor, normalized feature map whose shape is the same with input.
    """
    output = slim.batch_norm(x, decay=DECAY, center=center, scale=scale,
                             epsilon=EPSILON, updates_collections=tf.GraphKeys.UPDATE_OPS,
                             fused=False, is_training=is_train)

    return output


def instance_norm(x, center=True, scale=True, is_train=True):
    """Apply instance normalization.
    
    Args:
        x: tensor, input feature map.
        center: boolean, whether to use bias, i.e. beta.
        scale: bollean, whether to use affine parameters, i.e. \gamma.
        is_train: boolean, whether to update the buffers (moving average and var).

    Returns:
        tensor, normalized feature map whose shape is the same with input.
    """
    return slim.instance_norm(x, center=center, scale=scale, epsilon=EPSILON,
                              trainable=is_train)


def layer_norm(x, center=True, scale=True, is_train=True):
    """Apply layer normalization.
    
    Args:
        x: tensor, input feature map.
        center: boolean, whether to use bias, i.e. beta.
        scale: bollean, whether to use affine parameters, i.e. \gamma.
        is_train: boolean, whether to update the buffers (moving average and var).
    
    Returns:
        tensor, normalized feature map whose shape is the same with input.
    """
    return slim.layer_norm(x, center=True, scale=True, trainable=is_train)


NORM_FUNC = {
    "bn": batch_norm,
    "in": instance_norm,
    "ln": layer_norm,
}

class NormLayer(BaseLayer):
    """
    Normalization layer class.

    Args:
        norm_type: str, specifying the type of the norm layer. Possible choices:
            ('bn', 'ln', 'in').
        center: boolean, whether to use bias, i.e. beta.
        scale: bollean, whether to use affine parameters, i.e. \gamma.
        is_train: boolean, whether to update the buffers (moving average and var).

    Raises:
        ValueError, if the norm_type not in ('bn', 'ln', 'in')
    """
    def __init__(self, norm_type, center=True, scale=True, is_train=True):
        super(NormLayer, self).__init__()
        if norm_type not in NORM_FUNC:
            raise ValueError(f"Supported normalization layer type: {NORM_FUNC.keys()}, "
                             f"but is given {norm_type}")
        self.fn = NORM_FUNC[norm_type]
        self.is_train = is_train
        self.center = center
        self.scale = scale

    def forward(self, x):
        return self.fn(x, 
                       center=self.center, 
                       scale=self.scale, 
                       is_train=self.is_train)
