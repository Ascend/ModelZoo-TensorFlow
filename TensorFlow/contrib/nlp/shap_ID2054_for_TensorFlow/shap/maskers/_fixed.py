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
from ._masker import Masker

class Fixed(Masker):
    """ This leaves the input unchanged during masking, and is used for things like scoring labels.

    Sometimes there are inputs to the model that we do not want to explain, but rather we want to
    consider them fixed. The primary example of this is when we explain the loss of the model using
    the labels. These "true" labels are inputs to the function we are explaining, but we don't want
    to attribute credit to them, instead we want to consider them fixed and assign all the credit to
    the model's input features. This is where the Fixed masker can help, since we can apply it to the
    label inputs.
    """
    def __init__(self):
        self.shape = (None, 0)
        self.clustering = np.zeros((0, 4))

    def __call__(self, mask, x):
        return ([x],)

    def mask_shapes(self, x): # pylint: disable=no-self-use,unused-argument
        """ The shape of the masks we expect.
        """
        return [(0,)]

