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

training_params = {
    # the base learning rate used in the polynomial decay
    "base_lr":0.1,

    # how many steps to warmup the learning rate for
    "warmup_iter":3120,

    # What learning rate to start with in the warmup phase (ramps up to base_lr)
    "warmup_start_lr":0.025,

    #input size
    "image_size":227,

    # Block defs each tuple(x,y,z) describes one block with x number of filters at it's largest depth
    # y number of repeated units or bottlenecks, z stride for the first unit of the block.
    "block_defs":[(32,6,1),(64,6,2),(128,8,2),(256,1,2)],

    # definition of filters, kernel size and stride of the input convolution
    "input_def":(64,(7,7),2),

    # number of output classes
    "num_classes":1000,

    # How many groups to use for the grouped convolutions
    "groups": 1,

    # Whether to do relu before addition of the network and the residual
    "seperate_relus": 1
}