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
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
import numpy as np


def test_long_dense_vector():
    feature_columns = [SparseFeat('user_id', 4, ), SparseFeat('item_id', 5, ), DenseFeat("pic_vec", 5)]
    fixlen_feature_names = get_feature_names(feature_columns)

    user_id = np.array([[1], [0], [1]])
    item_id = np.array([[3], [2], [1]])
    pic_vec = np.array([[0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2]])
    label = np.array([1, 0, 1])

    input_dict = {'user_id': user_id, 'item_id': item_id, 'pic_vec': pic_vec}
    model_input = [input_dict[name] for name in fixlen_feature_names]

    model = DeepFM(feature_columns, feature_columns[:-1])
    model.compile('adagrad', 'binary_crossentropy')
    model.fit(model_input, label)


def test_feature_column_sparsefeat_vocabulary_path():
    vocab_path = "./dummy_test.csv"
    sf = SparseFeat('user_id', 4, vocabulary_path=vocab_path)
    if sf.vocabulary_path != vocab_path:
        raise ValueError("sf.vocabulary_path is invalid")
    vlsf = VarLenSparseFeat(sf, 6)
    if vlsf.vocabulary_path != vocab_path:
        raise ValueError("vlsf.vocabulary_path is invalid")

