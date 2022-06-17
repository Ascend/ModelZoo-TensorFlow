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
#from npu_bridge.npu_init import *
import pytest

try:
    from tensorflow.python.keras.utils import CustomObjectScope
except ImportError:
    from tensorflow.keras.utils import CustomObjectScope
from deepctr import layers
from tests.layers.interaction_test import BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE
from tests.utils import layer_test


@pytest.mark.parametrize(
    'axis',
    [-1, -2
     ]
)
def test_LayerNormalization(axis):
    with CustomObjectScope({'LayerNormalization': layers.LayerNormalization}):
        layer_test(layers.LayerNormalization, kwargs={"axis": axis, }, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))

