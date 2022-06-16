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
import tensorflow as tf

from deepctr.models.multitask import SharedBottom, ESMM, MMOE, PLE
from ..utils_mtl import get_mtl_test_data, check_mtl_model


def test_SharedBottom():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "SharedBottom"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = SharedBottom(dnn_feature_columns, bottom_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                         task_types=['binary', 'binary'], task_names=['label_income', 'label_marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_ESMM():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "ESMM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = ESMM(dnn_feature_columns, tower_dnn_hidden_units=(8,), task_types=['binary', 'binary'],
                 task_names=['label_marital', 'label_income'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_MMOE():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "MMOE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(8,),
                 tower_dnn_hidden_units=(8,),
                 gate_dnn_hidden_units=(), task_types=['binary', 'binary'],
                 task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


@pytest.mark.parametrize(
    'num_levels,gate_dnn_hidden_units',
    [(2, ()),
     (1, (4,))]
)
def test_PLE(num_levels, gate_dnn_hidden_units):
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "PLE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = PLE(dnn_feature_columns, num_levels=num_levels, expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                gate_dnn_hidden_units=gate_dnn_hidden_units,
                task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


if __name__ == "__main__":
    pass

