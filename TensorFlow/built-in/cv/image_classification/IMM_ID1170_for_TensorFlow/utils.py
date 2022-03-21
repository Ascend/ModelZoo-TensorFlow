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
import os
import time
import numpy as np


def SetDefaultAsNatural(FLAGS):
    if hasattr(FLAGS, 'epoch') and FLAGS.epoch < 0:
        FLAGS.epoch = 60

    if hasattr(FLAGS, 'learning_rate') and FLAGS.learning_rate < 0:
        FLAGS.learning_rate = 0.1

    if hasattr(FLAGS, 'regularizer') and FLAGS.regularizer < 0:
        FLAGS.regularizer = 1e-4

    if hasattr(FLAGS, 'alpha') and FLAGS.alpha < 0:
        FLAGS.alpha = 1.0 / 3

def PrintResults(alpha, results):
    """
    print accuracy results.

    Args:
        results: list of accuracy results.
            the half size of list is for training accuracy
            and the other is for test accuracy.
    """
    result_text = "%.2f" % alpha
    for i in range(int(len(results)/2)):
        result_text += ", train-idx%d: %.4f" % (i+1, results[i])
    for i in range(int(len(results)/2)):
        result_text += ", test-idx%d: %.4f" % (i+1, results[i + int(len(results)/2)])
    print(result_text)

