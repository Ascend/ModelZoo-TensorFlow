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
#==============================================================================

import atexit
from datetime import datetime
import json
from threading import Thread

_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None

def init(filename, run_name):
    global _file, _run_name
    _close_logfile()
    _file = open(filename, 'a')
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting new training run\n')
    _file.write('-----------------------------------------------------------------\n')
    _run_name = run_name


def log(msg, slack=False):
    print(msg)
    if _file is not None:
        _file.write('[%s]    %s\n' % (datetime.now().strftime(_format)[:-3], msg))

def _close_logfile():
    global _file
    if _file is not None:
        _file.close()
        _file = None

atexit.register(_close_logfile)
