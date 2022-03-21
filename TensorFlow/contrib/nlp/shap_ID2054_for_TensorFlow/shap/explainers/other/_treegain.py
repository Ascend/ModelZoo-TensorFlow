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
from npu_bridge.npu_init import *
from .._explainer import Explainer
import numpy as np

class TreeGain(Explainer):
    """ Simply returns the global gain/gini feature importances for tree models.

    This is only for benchmark comparisons and is not meant to approximate SHAP values.
    """
    def __init__(self, model):
        if str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            pass
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            pass
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            pass
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            pass
        else:
            raise Exception("The passed model is not yet supported by TreeGainExplainer: " + str(type(model)))
        assert hasattr(model, "feature_importances_"), "The passed model does not have a feature_importances_ attribute!"
        self.model = model

    def attributions(self, X):
        return np.tile(self.model.feature_importances_, (X.shape[0], 1))

