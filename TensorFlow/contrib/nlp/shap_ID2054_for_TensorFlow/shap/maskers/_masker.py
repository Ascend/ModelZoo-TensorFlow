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
from .._serializable import Serializable


class Masker(Serializable):
    """ This is the superclass of all maskers.
    """

    def __call__(self, mask, *args):
        """ Maskers are callable objects that accept the same inputs as the model plus a binary mask.
        """

    def _standardize_mask(self, mask, *args):
        """ This allows users to pass True/False as short hand masks.
        """
        if mask is True or mask is False:
            if callable(self.shape):
                shape = self.shape(*args)
            else:
                shape = self.shape

            if mask is True:
                return np.ones(shape[1], dtype=np.bool)
            return np.zeros(shape[1], dtype=np.bool)
        return mask

