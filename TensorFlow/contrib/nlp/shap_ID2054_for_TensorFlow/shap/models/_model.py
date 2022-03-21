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
import numpy as np
from .._serializable import Serializable, Serializer, Deserializer


class Model(Serializable):
    """ This is the superclass of all models.
    """

    def __init__(self, model=None):
        """ Wrap a callable model as a SHAP Model object.
        """
        if isinstance(model, Model):
            self.inner_model = model.inner_model
        else:
            self.inner_model = model

        if hasattr(model, "output_names"):
            self.output_names = model.output_names

    def __call__(self, *args):
        return np.array(self.inner_model(*args))

    def save(self, out_file):
        """ Save the model to the given file stream.
        """
        super().save(out_file)
        with Serializer(out_file, "shap.Model", version=0) as s:
            s.save("model", self.inner_model)

    @classmethod
    def load(cls, in_file, instantiate=True):
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Model", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model")
        return kwargs

