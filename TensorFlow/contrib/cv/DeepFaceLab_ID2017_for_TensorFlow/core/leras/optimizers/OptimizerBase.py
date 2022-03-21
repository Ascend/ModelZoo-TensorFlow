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

import copy
from core.leras import nn
tf = nn.tf

class OptimizerBase(nn.Saveable):
    def __init__(self, name=None):
        super().__init__(name=name)

    def tf_clip_norm(self, g, c, n):
        """Clip the gradient `g` if the L2 norm `n` exceeds `c`.
        # Arguments
            g: Tensor, the gradient tensor
            c: float >= 0. Gradients will be clipped
                when their L2 norm exceeds this value.
            n: Tensor, actual norm of `g`.
        # Returns
            Tensor, the gradient clipped if required.
        """
        if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
            return g

        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

        return g
nn.OptimizerBase = OptimizerBase
