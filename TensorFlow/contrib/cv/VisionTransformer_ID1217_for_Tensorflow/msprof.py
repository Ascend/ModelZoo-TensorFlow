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

#!/usr/bin/python3
# coding=utf-8
"""
Function:
This class mainly involves the main function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import importlib
import os
import sys

if __name__ == '__main__':
    sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
    _model_path = "ms_interface.msprof_entrance"
    MSPROF_ENTRANCE_CLASS = "MsprofEntrance"
    os.umask(0o027)
    msprof_entrance_module = importlib.import_module(_model_path)
    if hasattr(msprof_entrance_module, MSPROF_ENTRANCE_CLASS):
        getattr(msprof_entrance_module, MSPROF_ENTRANCE_CLASS)().main()
