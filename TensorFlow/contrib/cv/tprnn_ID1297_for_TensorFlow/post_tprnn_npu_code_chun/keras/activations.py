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
from . import backend as K
from .utils.generic_utils import get_from_module


def softmax(x):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor '
                         'that is not 2D or 3D. '
                         'Here, ndim=' + str(ndim))


def elu(x, alpha=1.0):
    return K.elu(x, alpha)


def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation function')
