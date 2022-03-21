# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
from npu_bridge.npu_init import *
import numpy as np

from ..utils import random_balanced_partitions


def test_random_balanced_partition():
    results = [
        random_balanced_partitions(np.array(['a', 'b', 'c']), 2, [3, 5, 5])
        for _ in range(100)
    ]
    results = [(a.tolist(), b.tolist()) for (a, b) in results]
    assert (['a', 'b'], ['c']) in results
    assert (['a', 'c'], ['b']) in results
    assert not (['b', 'c'], ['a']) in results

