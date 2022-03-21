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
""" Code for generating polynomials. """
import numpy as np
from npu_bridge.npu_init import *
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class PolyDataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        assert FLAGS.datasource == 'polynomial'
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        self.poly_order = FLAGS.poly_order

        self.generate = self.generate_polynomial_batch
        self.input_range = config.get('input_range', [-2.0, 2.0])
        self.coeff_range = config.get('coeff_range', [-1.0, 1.0])
        self.dim_input = 1
        self.dim_output = 1

    def generate_polynomial_batch(self):
        coeffs = np.random.uniform(self.coeff_range[0], self.coeff_range[1], [self.batch_size, self.poly_order+1])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        polynomial = np.polynomial.polynomial.polyval

        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(
                self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            func_coeffs = coeffs[func]  # [c0, c1,...,]
            for i in range(self.poly_order + 1):
                func_coeffs[i] /= (2 ** i)
            outputs[func] = polynomial(init_inputs[func], func_coeffs)

        return init_inputs, outputs


