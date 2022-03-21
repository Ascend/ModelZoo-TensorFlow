# Copyright 2022 Huawei Technologies Co., Ltd
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

__all__ = [
    'VISIBLE_PART',
    'MIN_NUM_JOINTS',
    'CENTER_TR',
    'SIGMA',
    'STRIDE',
    'SIGMA_CENTER',
    'INPUT_SIZE',
    'OUTPUT_SIZE',
    'NUM_JOINTS',
    'NUM_OUTPUT',
    'H36M_NUM_JOINTS',
    'JOINT_DRAW_SIZE',
    'LIMB_DRAW_SIZE'
]

# threshold
VISIBLE_PART = 1e-3
MIN_NUM_JOINTS = 5
CENTER_TR = 0.4

# net attributes
SIGMA = 7
STRIDE = 8
SIGMA_CENTER = 21
INPUT_SIZE = 368
OUTPUT_SIZE = 46
NUM_JOINTS = 14
NUM_OUTPUT = NUM_JOINTS + 1
H36M_NUM_JOINTS = 17

# draw options
JOINT_DRAW_SIZE = 3
LIMB_DRAW_SIZE = 1
NORMALISATION_COEFFICIENT = 1280*720

# test options
BATCH_SIZE = 4
