#
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
#



class AnyNetCfg:
    stem_type = "simple_stem_in"
    stem_w = 32
    block_type = "res_bottleneck_block"
    depths = []
    widths = []
    strides = []
    bot_muls = []
    group_ws = []
    se_on = True
    se_r = 0.25


class RegNetCfg:
    stem_type = "simple_stem_in"
    stem_w = 32
    block_type = "res_bottleneck_block"
    stride = 2
    se_on = True
    se_r = 0.25
    depth = 10
    w0 = 32
    wa = 5.0
    wm = 2.5
    group_w = 16
    bot_mul = 1.0
