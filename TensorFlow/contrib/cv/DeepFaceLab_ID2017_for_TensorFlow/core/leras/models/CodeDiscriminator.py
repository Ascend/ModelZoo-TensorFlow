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

class CodeDiscriminator(nn.ModelBase):
    def on_build(self, in_ch, code_res, ch=256, conv_kernel_initializer=None):            
        n_downscales = 1 + code_res // 8

        self.convs = []
        prev_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min( (2**i), 8 )
            self.convs.append ( nn.Conv2D( prev_ch, cur_ch, kernel_size=4 if i == 0 else 3, strides=2, padding='SAME', kernel_initializer=conv_kernel_initializer) )
            prev_ch = cur_ch

        self.out_conv =  nn.Conv2D( prev_ch, 1, kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)

    def forward(self, x):
        for conv in self.convs:
            x = tf.nn.leaky_relu( conv(x), 0.1 )
        return self.out_conv(x)
        
nn.CodeDiscriminator = CodeDiscriminator