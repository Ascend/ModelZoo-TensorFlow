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
from __future__ import absolute_import
from ..engine import Layer
from .. import backend as K
import numpy as np


class GaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        sigma: float, standard deviation of the noise distribution.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(GaussianNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianDropout(Layer):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        p: float, drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(p / (1 - p))`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, p, **kwargs):
        self.supports_masking = True
        self.p = p
        if 0 < p < 1:
            self.uses_learning_phase = True
        super(GaussianDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0 < self.p < 1:
            noise_x = x * K.random_normal(shape=K.shape(x), mean=1.0,
                                          std=np.sqrt(self.p / (1.0 - self.p)))
            return K.in_train_phase(noise_x, x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(GaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
