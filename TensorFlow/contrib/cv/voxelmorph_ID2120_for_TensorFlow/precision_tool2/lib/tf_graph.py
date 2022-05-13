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
import collections
import logging
import os
import config as cfg
from lib.util import util
from lib.precision_tool_exception import catch_tool_exception
from lib.precision_tool_exception import PrecisionToolException

CKPT_META_SHUFFIX = '.meta'


class TfGraph(object):
    def __init__(self, graph_root=cfg.TF_GRAPH_DIR):
        """"""
        self.graph_root = graph_root
        self.log = util.get_log()
        self.op_list = collections.OrderedDict()

    @catch_tool_exception
    def get_op_list(self, ckpt_path=None):
        if self.op_list is None:
            self._convert_ckpt_to_graph(ckpt_path)
        return self.op_list

    def _convert_ckpt_to_graph(self, ckpt_path):
        log_level = self.log.level
        try:
            self.log.setLevel('ERROR')
            import tensorflow as tf
            self.log.setLevel(log_level)
        except ImportError as err:
            self.log.setLevel(log_level)
            raise PrecisionToolException("Import tensorflow failed.")
        meta_files = util.list_cpu_graph_files(ckpt_path)
        if len(meta_files) == 0:
            raise PrecisionToolException("Can not find any ckpt meta files.")
        file_list = sorted(meta_files.values(), key=lambda x: x['timestamp'])
        ckpt_file = file_list[-1]
        self.log.info("Find %d tf ckpt meta files, choose [%s]" % (
            len(meta_files), ckpt_file['file_name']))
        self.op_list = collections.OrderedDict()
        saver = tf.train.import_meta_graph(
            ckpt_file['path'], clear_devices=True)
        graph = tf.get_default_graph()
        for op in graph.get_operations():
            self.op_list[op.name] = op
