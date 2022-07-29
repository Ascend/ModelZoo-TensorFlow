# encoding = utf-8
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
def test_ls():
    np.testing.assert_equal(ls('.', suffix = '.py'), ls('.', suffix = '.PY'))

@util.dec.print_test
def test_readlines():
    p = __file__
    lines = read_lines(p)
    for l in lines:
        print(l)    

@util.dec.print_test
def test_write_lines():
    p = '~/temp/log/w.txt'
    lines = read_lines(__file__)
    write_lines(p, lines)
    lines2 = read_lines(p)
    np.testing.assert_equal(lines, lines2)
    
@util.dec.print_test
def test_mat_io():
    path = '~/temp/testpython.mat'
    util.io.dump_mat(path, {'a': 1,'b': 2,'c': 3,'d': np.ones((3, 3))})   
    data =  util.io.load_mat(path)
    for c in data:
        print(data[c])
      
    vs = util.io.dir_mat(path)
    print(vs)
#test_ls()
#test_readlines()
#test_write_lines()
test_mat_io()


