# encoding=utf-8
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
from npu_bridge.npu_init import *
from common_import import *

@util.dec.print_test
def test_normal():
    shape = (500, 500)
    mu = 10
    sigma_square = 20
    a = util.rand.normal(shape = shape, mu = mu, sigma_square = sigma_square)
    E = util.statistic.E(a)
    D = util.statistic.D(a)
    np.testing.assert_almost_equal(E, mu, 0)
    np.testing.assert_almost_equal(D, sigma_square, 0)

@util.dec.print_test
def test_randint():
#    logging.info('generating random int:%d'%(util.rand.randint()))
    print(util.rand.randint())
    print(util.rand.randint(10))
    print(util.rand.randint(shape = (2, 3)))
    
@util.dec.print_test
def test_sample():
    lst = range(1000)
    n = 20
    samples = util.rand.sample(zip(lst, lst), n)
    util.test.assert_equal(len(samples), n)

#test_randint()
#test_normal()
test_sample()

