#!/usr/bin/env python
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
"""
Copyright 2017, Zixin Luo, HKUST.
Commonly used functions
"""

from __future__ import print_function
from npu_bridge.npu_init import *

import os
from datetime import datetime


class ClassProperty(property):
    """For dynamically obtaining system time"""
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


class Notify(object):
    """Colorful printing prefix.
    A quick example:
    print(Notify.INFO, YOUR TEXT, Notify.ENDC)
    """

    def __init__(self):
        pass

    @ClassProperty
    def HEADER(cls):
        return str(datetime.now()) + ': \033[95m'

    @ClassProperty
    def INFO(cls):
        return str(datetime.now()) + ': \033[92mI'

    @ClassProperty
    def OKBLUE(cls):
        return str(datetime.now()) + ': \033[94m'

    @ClassProperty
    def WARNING(cls):
        return str(datetime.now()) + ': \033[93mW'

    @ClassProperty
    def FAIL(cls):
        return str(datetime.now()) + ': \033[91mF'

    @ClassProperty
    def BOLD(cls):
        return str(datetime.now()) + ': \033[1mB'

    @ClassProperty
    def UNDERLINE(cls):
        return str(datetime.now()) + ': \033[4mU'
    ENDC = '\033[0m'


def read_list(list_path):
    """Read list."""
    if list_path is None or not os.path.exists(list_path):
        print(Notify.FAIL, 'Not exist', list_path, Notify.ENDC)
        exit()
    content = open(list_path).read().splitlines()
    return content


def write_list(list_in, path_save):
    """Write list."""
    fout = open(path_save, 'w')
    fout.write('\n'.join(list_in))


def replace_str_in_file(list_in, orig_str, dest_str):
    """Replace strings in a file."""
    if os.path.exists(list_in):
        content = open(list_in).read()
        new_content = content.replace(orig_str, dest_str)
        open(list_in, 'w').write(new_content)
    else:
        print(Notify.WARNING + 'Not exist', list_in, Notify.ENDC)

