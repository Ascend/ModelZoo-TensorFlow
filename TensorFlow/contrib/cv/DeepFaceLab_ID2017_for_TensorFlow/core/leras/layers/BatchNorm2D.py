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

from core.leras import nn
tf = nn.tf

class BatchNorm2D(nn.LayerBase):
    """
    currently not for training
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        self.weight       = tf.get_variable("weight",   (self.dim,), dtype=self.dtype, initializer=tf.initializers.ones() )
        self.bias         = tf.get_variable("bias",     (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros() )
        self.running_mean = tf.get_variable("running_mean", (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros(), trainable=False )
        self.running_var  = tf.get_variable("running_var",  (self.dim,), dtype=self.dtype, initializer=tf.initializers.zeros(), trainable=False )

    def get_weights(self):
        return [self.weight, self.bias, self.running_mean, self.running_var]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.dim)
        else:
            shape = (1,self.dim,1,1)

        weight       = tf.reshape ( self.weight      , shape )
        bias         = tf.reshape ( self.bias        , shape )
        running_mean = tf.reshape ( self.running_mean, shape )
        running_var  = tf.reshape ( self.running_var , shape )

        x = (x - running_mean) / tf.sqrt( running_var + self.eps )
        x *= weight
        x += bias
        return x

nn.BatchNorm2D = BatchNorm2D