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

class Coefficent(Explainer):
    """ Simply returns the model coefficents as the feature attributions.

    This is only for benchmark comparisons and does not approximate SHAP values in a
    meaningful way.
    """
    def __init__(self, model):
        assert hasattr(model, "coef_"), "The passed model does not have a coef_ attribute!"
        self.model = model

    def attributions(self, X):
        return np.tile(self.model.coef_, (X.shape[0], 1))

