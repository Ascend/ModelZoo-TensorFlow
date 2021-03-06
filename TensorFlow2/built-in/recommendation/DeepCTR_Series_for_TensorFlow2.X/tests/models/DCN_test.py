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

from deepctr.estimator import DCNEstimator
from deepctr.models import DCN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    Estimator_TEST_TF1


@pytest.mark.parametrize(
    'cross_num,hidden_size,sparse_feature_num,cross_parameterization',
    [(0, (8,), 2, 'vector'), (1, (), 1, 'vector'), (1, (8,), 3, 'vector'),
     (0, (8,), 2, 'matrix'), (1, (), 1, 'matrix'), (1, (8,), 3, 'matrix'),
     ]
)
def test_DCN(cross_num, hidden_size, sparse_feature_num, cross_parameterization):
    model_name = "DCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DCN(feature_columns, feature_columns, cross_num=cross_num, cross_parameterization=cross_parameterization,
                dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    check_model(model, model_name, x, y)


def test_DCN_2():
    model_name = "DCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=3,
                                          dense_feature_num=2)

    model = DCN([], feature_columns, cross_num=1, dnn_hidden_units=(8,), dnn_dropout=0.5)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'cross_num,hidden_size,sparse_feature_num',
    [(1, (8,), 3)
     ]
)
def test_DCNEstimator(cross_num, hidden_size, sparse_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = DCNEstimator(linear_feature_columns, dnn_feature_columns, cross_num=cross_num, dnn_hidden_units=hidden_size,
                         dnn_dropout=0.5)
    check_estimator(model, input_fn)


# def test_DCN_invalid(embedding_size=8, cross_num=0, hidden_size=()):
#     feature_dim_dict = {'sparse': [SparseFeat('sparse_1', 2), SparseFeat('sparse_2', 5), SparseFeat('sparse_3', 10)],
#                         'dense': [SparseFeat('dense_1', 1), SparseFeat('dense_1', 1), SparseFeat('dense_1', 1)]}
#     with pytest.raises(ValueError):
#         _ = DCN(None, embedding_size=embedding_size, cross_num=cross_num, dnn_hidden_units=hidden_size, dnn_dropout=0.5)


if __name__ == "__main__":
    pass

