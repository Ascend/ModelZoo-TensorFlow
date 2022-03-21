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

class ScaleAdd(nn.LayerBase):
    def __init__(self, ch, dtype=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        self.ch = ch

        super().__init__(**kwargs)

    def build_weights(self):
        self.weight = tf.get_variable("weight",(self.ch,), dtype=self.dtype, initializer=tf.initializers.zeros() )

    def get_weights(self):
        return [self.weight]

    def forward(self, inputs):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.ch)
        else:
            shape = (1,self.ch,1,1)

        weight = tf.reshape ( self.weight, shape )

        x0, x1 = inputs
        x = x0 + x1*weight

        return x
nn.ScaleAdd = ScaleAdd