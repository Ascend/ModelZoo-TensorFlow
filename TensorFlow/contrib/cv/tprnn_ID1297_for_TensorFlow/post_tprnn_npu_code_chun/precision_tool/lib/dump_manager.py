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
import collections
from lib.util import util
from lib.constant import Constant
from lib.npu_dump import NpuDump
from lib.tf_dump import TfDump
import config as cfg
from lib.precision_tool_exception import catch_tool_exception
from lib.precision_tool_exception import PrecisionToolException


class DumpManager(object):
    def __init__(self):
        self.npu_dumps = collections.OrderedDict()
        self.tf_dump = TfDump(cfg.TF_DUMP_DIR)
        self._init_dirs()

    def prepare(self):
        # prepare npu
        sub_dirs = os.listdir(cfg.NPU_DIR)
        if len(sub_dirs) == 0:
            # create default
            sub_dirs = [Constant.DEFAULT_DEBUG_ID]
        sorted(sub_dirs)
        for sub_dir in sub_dirs:
            npu_dump = NpuDump(sub_dir)
            npu_dump.prepare()
            self.npu_dumps[sub_dir] = npu_dump
        # prepare tf
        self.tf_dump.prepare()

    def get_dump_root_dir(self, debug_id):
        if debug_id in self.npu_dumps:
            return self.npu_dumps[debug_id].dump_root
        return None

    def op_dump_summary(self, ops):
        npu_result = collections.OrderedDict()
        for debug_id, op in ops.items():
            if debug_id in self.npu_dumps:
                npu_result[debug_id] = collections.OrderedDict()
                for op_detail in op:
                    npu_result[debug_id][op_detail.graph_name] = self.npu_dumps[
                        debug_id
                    ].op_dump_summary(op_detail)
        tf_result = None
        if self.tf_dump is not None and len(ops[Constant.DEFAULT_DEBUG_ID]) != 0:
            tf_result = self.tf_dump.op_dump_summary(ops[Constant.DEFAULT_DEBUG_ID][0])
        return npu_result, tf_result

    def print_tensor(self, file_name, is_convert):
        """Print numpy data file"""
        if os.path.isfile(file_name):
            return util.print_npy_summary(
                os.path.dirname(file_name), os.path.basename(file_name), is_convert
            )
        file_name = file_name.replace("/", "_")
        npu_convert_files = util.list_npu_dump_convert_files(cfg.DECODE_DIR, file_name)
        tf_decode_files = util.list_cpu_dump_decode_files(cfg.TF_DUMP_DIR, file_name)
        self._print_tensors(npu_convert_files, is_convert)
        self._print_tensors(tf_decode_files, is_convert)

    @staticmethod
    def _print_tensors(file_infos, is_convert):
        if file_infos is not None:
            for file_info in file_infos.values():
                util.print_npy_summary(
                    file_info.dir_path, file_info.file_name, is_convert
                )

    @staticmethod
    def _init_dirs():
        """Create dump file dirs"""
        util.create_dir(cfg.DUMP_DECODE_DIR)
        util.create_dir(cfg.NPU_OVERFLOW_DUMP_DIR)
        util.create_dir(cfg.OVERFLOW_DECODE_DIR)
        util.create_dir(cfg.TF_DUMP_DIR)
