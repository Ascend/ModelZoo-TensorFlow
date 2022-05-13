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
''' initializations for the neuron project '''

# general imports
import os
import numpy as np
import tensorflow.python.keras.backend as K


def output_init(shape, name=None, dim_ordering=None):
    ''' initialization for output weights'''
    size = (shape[0], shape[1], shape[2] - shape[3], shape[3])

    # initialize output weights with random and identity
    rpart = np.random.random(size)
#     idpart_ = np.eye(size[3])
    idpart_ = np.ones((size[3], size[3]))
    idpart = np.expand_dims(np.expand_dims(idpart_, 0), 0)
    value = np.concatenate((rpart, idpart), axis=2)
    return K.variable(value, name=name)
