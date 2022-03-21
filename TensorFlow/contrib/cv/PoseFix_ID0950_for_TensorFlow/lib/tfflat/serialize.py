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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
### modified from https://github.com/ppwwyyxx/tensorpack

from npu_bridge.npu_init import *
import os
import sys
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

# https://github.com/apache/arrow/pull/1223#issuecomment-359895666
old_mod = sys.modules.get('torch', None)
sys.modules['torch'] = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
if old_mod is not None:
    sys.modules['torch'] = old_mod
else:
    del sys.modules['torch']

import pickle

__all__ = ['loads', 'dumps', 'dump_pkl', 'load_pkl']


def dumps_msgpack(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, raw=False)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def dump_pkl(name, obj):
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL )

def load_pkl(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        ret = pickle.load( f )
    return ret

if pa is None:
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow


