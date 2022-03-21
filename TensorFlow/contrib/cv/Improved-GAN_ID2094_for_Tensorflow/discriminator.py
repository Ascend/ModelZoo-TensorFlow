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
from npu_bridge.npu_init import *
import numpy as np
import tensorflow as tf
from ops import conv2d
from util import log


class Discriminator(object):
    def __init__(self, name, num_class, norm_type, is_train):
        self.name = name
        self._num_class = num_class
        self._norm_type = norm_type
        self._is_train = is_train
        self._reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = input
            num_channel = [32, 64, 128, 256, 256, 512]
            num_layer = np.ceil(np.log2(min(_.shape.as_list()[1:3]))).astype(np.int)
            for i in range(num_layer):
                ch = num_channel[i] if i < len(num_channel) else 512
                _ = conv2d(_, ch, self._is_train, info=not self._reuse,
                           norm=self._norm_type, name='conv{}'.format(i+1))
            _ = conv2d(_, int(num_channel[i]/4), self._is_train, k=1, s=1,
                       info=not self._reuse, norm='None', name='conv{}'.format(i+2))
            _ = conv2d(_, self._num_class+1, self._is_train, k=1, s=1, info=not self._reuse,
                       activation_fn=None, norm='None',
                       name='conv{}'.format(i+3))
            _ = tf.squeeze(_)
            if not self._reuse: 
                log.info('discriminator output {}'.format(_.shape.as_list()))
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return tf.nn.sigmoid(_), _

