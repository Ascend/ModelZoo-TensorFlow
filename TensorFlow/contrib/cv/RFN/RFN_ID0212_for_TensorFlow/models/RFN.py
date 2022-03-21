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


import tensorflow as tf
import math
from . import block as B


def RFN(rgba, nf, nb, out_nc, upscale=4, act_type='lrelu'):
    """Architecture of RFN model"""

    n_upscale = int(math.log(upscale, 2))
    if upscale == 3:
        n_upscale = 1

    c1 = B.conv_layer(rgba, filters=nf, kernel_size=3)
    net = tf.identity(c1)
    for _ in range(nb):
        net = B.RRBlock_32(net)
    net = B.conv_layer(net, filters=nf, kernel_size=3)
    net += c1

    if upscale == 3:
        net = B.upconv_block(net, out_channels=nf,
                             upscale_factor=3, act_type=act_type)
    else:
        for _ in range(n_upscale):
            net = B.upconv_block(net, out_channels=nf, act_type=act_type)
    net = B.conv_block(net, out_nc=nf, kernel_size=3,
                       norm_type=None, act_type=act_type)
    net = B.conv_block(net, out_nc=out_nc, kernel_size=3,
                       norm_type=None, act_type=None)

    return net
