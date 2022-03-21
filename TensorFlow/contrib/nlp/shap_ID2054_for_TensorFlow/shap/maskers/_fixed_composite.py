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
from ._masker import Masker
from .._serializable import Serializer, Deserializer


class FixedComposite(Masker):
    """ A masker that outputs both the masked data and the original data as a pair.
    """

    def __init__(self, masker):
        """ Creates a Composite masker from an underlying masker and returns the original args along with the masked output.

        Parameters
        ----------
        masker: object
            An object of the shap.maskers.Masker base class (eg. Text/Image masker).

        Returns
        -------
        tuple
            A tuple consisting of the masked input using the underlying masker appended with the original args in a list.
        """
        self.masker = masker

        # copy attributes from the masker we are wrapping
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names", "text_data", "image_data"]
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Computes mask on the args using the masker data attribute and returns tuple containing masked input with args.
        """
        masked_X = self.masker(mask, *args)
        wrapped_args = []
        for item in args:
            wrapped_args.append(np.array([item]))
        wrapped_args = tuple(wrapped_args)
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + wrapped_args

    def save(self, out_file):
        """ Write a FixedComposite masker to a file stream.
        """
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.maskers.FixedComposite", version=0) as s:
            s.save("masker", self.masker)

    @classmethod
    def load(cls, in_file, instantiate=True):
        """ Load a FixedComposite masker from a file stream.
        """
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.maskers.FixedComposite", min_version=0, max_version=0) as s:
            kwargs["masker"] = s.load("masker")
        return kwargs

