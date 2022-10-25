# encoding:utf-8
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
from npu_bridge.npu_init import *
import numpy as np


class Config:
    MEAN = np.float32([102.9801, 115.9465, 122.7717])
    # MEAN=np.float32([100.0, 100.0, 100.0])
    TEST_GPU_ID = 0
    SCALE = 900
    MAX_SCALE = 1500
    TEXT_PROPOSALS_WIDTH = 0
    MIN_RATIO = 0.01
    LINE_MIN_SCORE = 0.6
    TEXT_LINE_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 30
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_NUM_PROPOSALS = 0
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6
