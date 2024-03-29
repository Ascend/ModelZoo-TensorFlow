from npu_bridge.npu_init import *
import pytest

from deepctr.models import DeepFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, TEST_Estimator


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 1),  #
     ((3,), 2)
     ]  # (True, (32,), 3), (False, (32,), 1)
)
def test_DeepFM(hidden_size, sparse_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DeepFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [
        ((3,), 2)
    ]  # (True, (32,), 3), (False, (32,), 1)
)
def test_DeepFMEstimator(hidden_size, sparse_feature_num):
    if not TEST_Estimator:
        return
    from deepctr.estimator import DeepFMEstimator
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num,
                                                                                    classification=False)

    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5,
                            task="regression")

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass

