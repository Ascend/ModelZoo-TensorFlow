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
import pytest
import tensorflow as tf

from deepctr.estimator import FwFMEstimator
from deepctr.models import FwFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    Estimator_TEST_TF1


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 1),
     ((), 1),
     ]
)
def test_FwFM(hidden_size, sparse_feature_num):
    model_name = "FwFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = FwFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 2),
     ]
)
def test_FwFMEstimator(hidden_size, sparse_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = FwFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass

