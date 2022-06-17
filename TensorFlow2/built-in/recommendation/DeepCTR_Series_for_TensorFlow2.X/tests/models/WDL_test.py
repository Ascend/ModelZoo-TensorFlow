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
from packaging import version

from deepctr.estimator import WDLEstimator
from deepctr.models import WDL
from ..utils import check_model, check_estimator, SAMPLE_SIZE, get_test_data, get_test_data_estimator, \
    Estimator_TEST_TF1


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2, 0), (0, 2)  # ,(2, 2)
     ]
)
def test_WDL(sparse_feature_num, dense_feature_num):
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        return
    model_name = "WDL"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num, hash_flag=True)

    model = WDL(feature_columns, feature_columns,
                dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2, 1),  # (0, 2)#,(2, 2)
     ]
)
def test_WDLEstimator(sparse_feature_num, dense_feature_num):
    if not Estimator_TEST_TF1 and version.parse(tf.__version__) < version.parse('2.2.0'):
        return
    sample_size = SAMPLE_SIZE

    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size, sparse_feature_num,
                                                                                    dense_feature_num)
    model = WDLEstimator(linear_feature_columns, dnn_feature_columns,
                         dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass

