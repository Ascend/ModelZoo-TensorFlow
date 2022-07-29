#encoding = utf-8
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
def test_to_lowercase():
    np.testing.assert_string_equal(util.str.to_lowercase('Ss'), 'ss')
    

@util.dec.print_test
def test_endswith():
    np.testing.assert_(util.str.ends_with('hello.ss', 'ss'))
    np.testing.assert_(util.str.ends_with('hello.ss', '.SS', ignore_case = True))
    np.testing.assert_(util.str.ends_with('hello.ss', ['ss', 'SS']))

@util.dec.print_test
def test_startswith():
    np.testing.assert_(util.str.starts_with('hello.ss', 'he'))
    np.testing.assert_(util.str.starts_with('hello.ss', 'HeL', ignore_case = True))
    np.testing.assert_(util.str.starts_with('hello.ss', ['h', 'SS']))

    
@util.dec.print_test
def test_is_str():
    np.testing.assert_(util.str.is_str(''))
    np.testing.assert_(not util.str.is_str([]))
    np.testing.assert_(not util.str.is_str(0))

@util.dec.print_test
def test_contains():
    s = 'This is China'
    target = 'this'
    np.testing.assert_(not util.str.contains(s, target, ignore_case = False))
    np.testing.assert_(util.str.contains(s, target, ignore_case = True))


def test_replace_all():
    s = 'a  \t b\t  c'
    r = util.str.replace_all(s, ' ', '')
    r = util.str.replace_all(r, '\t', '')    
    np.testing.assert_equal(r, 'abc')

if util.mod.is_main(__name__):
    #test_to_lowercase()
    #test_endswith()
    #test_is_str()
    #test_contains()
    #test_replace_all()
    test_startswith()

