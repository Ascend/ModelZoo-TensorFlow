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
from ops import fc


class Generator(object):
    def __init__(self, name, h, w, c, norm_type, deconv_type, is_train):
        self.name = name
        self._h = h
        self._w = w
        self._c = c
        self._norm_type = norm_type
        self._deconv_type = deconv_type
        self._is_train = is_train
        self._reuse = False

    def __call__(self, input):
        if self._deconv_type == 'bilinear':
            from ops import bilinear_deconv2d as deconv2d
        elif self._deconv_type == 'nn':
            from ops import nn_deconv2d as deconv2d
        elif self._deconv_type == 'transpose':
            from ops import deconv2d
        else:
            raise NotImplementedError
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = tf.reshape(input, [input.get_shape().as_list()[0], 1, 1, -1])
            _ = fc(_, 1024, self._is_train, info=not self._reuse, norm='None', name='fc')
            for i in range(int(np.ceil(np.log2(max(self._h, self._w))))):
                _ = deconv2d(_, max(self._c, int(_.get_shape().as_list()[-1]/2)), 
                             self._is_train, info=not self._reuse, norm=self._norm_type,
                             name='deconv{}'.format(i+1))
            _ = deconv2d(_, self._c, self._is_train, k=1, s=1, info=not self._reuse,
                         activation_fn=tf.tanh, norm='None',
                         name='deconv{}'.format(i+2))
            _ = tf.image.resize_bilinear(_, [self._h, self._w])

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _

