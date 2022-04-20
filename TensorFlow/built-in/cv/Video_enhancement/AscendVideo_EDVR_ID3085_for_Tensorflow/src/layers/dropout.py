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

from src.layers.base_layer import BaseLayer
from src.utils.logger import logger

try:
    from npu_bridge.estimator import npu_ops
    OP_IMPL = 'npu'
except Exception:
    logger.error('Failed to import NPU dropout. Please use the composed tf operator instead.')
    OP_IMPL = tf


__all__ = ["Dropout"]

class Dropout(BaseLayer):
    """Dropout layer. 
    
    Use NPU high performance operator if possible.

    Args:
        keep_prob: float, ranged in [0, 1], specifying the keeping probability
            of the feature point.
    """
    def __init__(self, keep_prob=0.1, name=None):
        self.name = name
        self.keep_prob = keep_prob

    def forward(self, input_tensor, training=False):
        """Perform dropout.
        """
        if not training:
            return input_tensor

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if OP_IMPL == tf:
                output = tf.nn.dropout(input_tensor, self.keep_prob)
            else:
                if self.keep_prob is None or self.keep_prob == 1.0:
                    return input_tensor

                ##################modify for npu######################
                # Modify dropout for high performance
                output = npu_ops.dropout(input_tensor, self.keep_prob)
                ##################npu modify end######################
            return output
