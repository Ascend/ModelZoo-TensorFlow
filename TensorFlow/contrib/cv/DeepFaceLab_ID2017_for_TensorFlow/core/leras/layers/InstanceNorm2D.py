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

class InstanceNorm2D(nn.LayerBase):
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)
        self.weight       = tf.get_variable("weight",   (self.in_ch,), dtype=self.dtype, initializer=kernel_initializer )
        self.bias         = tf.get_variable("bias",     (self.in_ch,), dtype=self.dtype, initializer=tf.initializers.zeros() )

    def get_weights(self):
        return [self.weight, self.bias]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.in_ch)
        else:
            shape = (1,self.in_ch,1,1)

        weight       = tf.reshape ( self.weight      , shape )
        bias         = tf.reshape ( self.bias        , shape )

        x_mean = tf.reduce_mean(x, axis=nn.conv2d_spatial_axes, keepdims=True )
        x_std  = tf.math.reduce_std(x, axis=nn.conv2d_spatial_axes, keepdims=True ) + 1e-5

        x = (x - x_mean) / x_std
        x *= weight
        x += bias

        return x

nn.InstanceNorm2D = InstanceNorm2D