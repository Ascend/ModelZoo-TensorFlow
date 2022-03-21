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

import numpy as np
from core.leras import nn
tf = nn.tf

class BlurPool(nn.LayerBase):
    def __init__(self, filt_size=3, stride=2, **kwargs ):

        if nn.data_format == "NHWC":
            self.strides = [1,stride,stride,1]
        else:
            self.strides = [1,1,stride,stride]

        self.filt_size = filt_size
        pad = [ int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ]

        if nn.data_format == "NHWC":
            self.padding = [ [0,0], pad, pad, [0,0] ]
        else:
            self.padding = [ [0,0], [0,0], pad, pad ]

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:,None]*a[None,:]
        a = a / np.sum(a)
        a = a[:,:,None,None]
        self.a = a
        super().__init__(**kwargs)

    def build_weights(self):
        self.k = tf.constant (self.a, dtype=nn.floatx )

    def forward(self, x):
        k = tf.tile (self.k, (1,1,x.shape[nn.conv2d_ch_axis],1) )
        x = tf.pad(x, self.padding )
        x = tf.nn.depthwise_conv2d(x, k, self.strides, 'VALID', data_format=nn.data_format)
        return x
nn.BlurPool = BlurPool