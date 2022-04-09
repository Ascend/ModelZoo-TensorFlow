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
import pytest
import tensorflow as tf

from deepctr.layers.utils import Hash, Linear
from tests.layers.interaction_test import BATCH_SIZE, EMBEDDING_SIZE
from tests.utils import layer_test

try:
    from tensorflow.python.keras.utils import CustomObjectScope
except ImportError:
    from tensorflow.keras.utils import CustomObjectScope


@pytest.mark.parametrize(
    'num_buckets,mask_zero,vocabulary_path,input_data,expected_output',
    [
        (3 + 1, False, None, ['lakemerson'], None),
        (3 + 1, True, None, ['lakemerson'], None),
        (
                3 + 1, False, "./tests/layers/vocabulary_example.csv", [['lake'], ['johnson'], ['lakemerson']],
                [[1], [3], [0]])
    ]
)
def test_Hash(num_buckets, mask_zero, vocabulary_path, input_data, expected_output):
    if not hasattr(tf, 'version') or tf.version.VERSION < '2.0.0':
        return

    with CustomObjectScope({'Hash': Hash}):
        layer_test(Hash,
                   kwargs={'num_buckets': num_buckets, 'mask_zero': mask_zero, 'vocabulary_path': vocabulary_path},
                   input_dtype=tf.string, input_data=np.array(input_data, dtype='str'),
                   expected_output_dtype=tf.int64, expected_output=expected_output)


def test_Linear():
    with CustomObjectScope({'Linear': Linear}):
        layer_test(Linear,
                   kwargs={'mode': 1, 'use_bias': True}, input_shape=(BATCH_SIZE, EMBEDDING_SIZE))

