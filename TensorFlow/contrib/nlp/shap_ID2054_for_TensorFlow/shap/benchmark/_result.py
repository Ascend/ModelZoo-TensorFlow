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
from npu_bridge.npu_init import *
import numpy as np
import sklearn

sign_defaults = {
    "keep positive": 1,
    "keep negative": -1,
    "remove positive": -1,
    "remove negative": 1,
    "compute time": -1,
    "keep absolute": -1, # the absolute signs are defaults that make sense when scoring losses
    "remove absolute": 1,
    "explanation error": -1
}

class BenchmarkResult():
    """ The result of a benchmark run.
    """

    def __init__(self, metric, method, value=None, curve_x=None, curve_y=None, value_sign=None):
        self.metric = metric
        self.method = method
        self.value = value
        self.curve_x = curve_x
        self.curve_y = curve_y
        self.value_sign = value_sign
        if self.value_sign is None and self.metric in sign_defaults:
            self.value_sign = sign_defaults[self.metric]
        if self.value is None:
            self.value = sklearn.metrics.auc(curve_x, (np.array(curve_y) - curve_y[0]))

    @property
    def full_name(self):
        return self.method + " " + self.metric

