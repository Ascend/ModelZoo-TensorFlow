# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import random
import time


class Base:
    """Base class for degradation.
    """
    def set_numpy_random_seed(self):
        # Numpy random states are the same across all the mutli-processing.
        # In order to maintain the randomness, use the system timestamp to
        # manually set numpy seed every time this function is called.
        np.random.seed(int(time.time() + random.random() * 1000000))

    def check_input(self, x):
        return isinstance(x, np.ndarray) and x.ndim == 3

    def __call__(self, im):
        self.set_numpy_random_seed()
        if self.check_input(im):
            return self.apply(im)
        else:
            raise ValueError(f'Expect input image to be 3D-array (HWC), but got {im.ndim}D-array.')

    def apply(self, im):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}@{id(self)}'

    def __str__(self):
        return self.__repr__()