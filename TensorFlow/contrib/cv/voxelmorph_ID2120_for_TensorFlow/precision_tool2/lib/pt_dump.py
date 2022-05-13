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

# coding=utf-8
import os
import re
import time
import sys
from lib.util import util
from lib.constant import Constant
from lib.h5_util import H5Util
import config as cfg
from lib.precision_tool_exception import catch_tool_exception
from lib.precision_tool_exception import PrecisionToolException


class PtDump(object):
    def __init__(self, data_dir):
        self.log = util.get_log()
        self.npu = None
        self.gpu = None
        self.data_dir = data_dir

    def prepare(self):
        util.create_dir(cfg.PT_NPU_DIR)
        util.create_dir(cfg.PT_GPU_DIR)
        if not util.empty_dir(cfg.PT_NPU_DIR):
            self.npu = H5Util()
        if not util.empty_dir(cfg.PT_GPU_DIR):
            self.gpu = H5Util()

    def get_dump_files_by_name(self, file_name):
        """Get dump files by name"""
        print(file_name)
        return {}
