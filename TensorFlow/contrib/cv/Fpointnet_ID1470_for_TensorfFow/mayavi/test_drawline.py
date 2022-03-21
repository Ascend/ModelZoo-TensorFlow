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

import numpy
from mayavi.mlab import *

def test_plot3d():
    """Generates a pretty set of lines."""
    n_mer, n_long = 6, 11
    pi = numpy.pi
    dphi = pi / 1000.0
    phi = numpy.arange(0.0, 2 * pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x = numpy.cos(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    y = numpy.sin(mu) * (1 + numpy.cos(n_long * mu / n_mer) * 0.5)
    z = numpy.sin(n_long * mu / n_mer) * 0.5

    l = plot3d(x, y, z, numpy.sin(mu), tube_radius=0.025, colormap='Spectral')
    return l

test_plot3d()
raw_input()
