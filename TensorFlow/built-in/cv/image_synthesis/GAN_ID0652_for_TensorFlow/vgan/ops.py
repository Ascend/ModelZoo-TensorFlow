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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import tensorflow as tf
import tensorflow.contrib.slim as slim


def flatten_fully_connected(inputs,
                            num_outputs,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):
    with tf.variable_scope(scope, 'flatten_fully_connected', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)


def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y
