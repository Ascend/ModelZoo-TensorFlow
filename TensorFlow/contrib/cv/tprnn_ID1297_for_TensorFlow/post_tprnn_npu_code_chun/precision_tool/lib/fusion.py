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
import json
import os
import shutil

from lib.tool_object import ToolObject
from lib.util import util
import config as cfg

FUSION_RESULT_FILE_NAME = "fusion_result.json"
EFFECT_TIMES_KEY = "effect_times"
GRAPH_FUSION_KEY = "graph_fusion"
UB_FUSION_KEY = "ub_fusion"
GRAPH_ID_KEYS = ["graphId", "session_and_graph_id"]


class FusionResult(object):
    def __init__(self, fusion_json):
        self.fusion_json = fusion_json

    def get_effect_graph_fusion(self):
        """Get effect graph fusion rule"""
        if GRAPH_FUSION_KEY in self.fusion_json:
            return self._get_effect_fusion(self.fusion_json[GRAPH_FUSION_KEY])
        return {}

    def get_effect_ub_fusion(self):
        """Get effect UB fusion rule"""
        if UB_FUSION_KEY in self.fusion_json:
            return self._get_effect_fusion(self.fusion_json[UB_FUSION_KEY])
        return {}

    def graph_id(self):
        """Get graph id"""
        for key in GRAPH_ID_KEYS:
            if key in self.fusion_json:
                return self.fusion_json[key]
        return "NONE"

    @staticmethod
    def _get_effect_fusion(fusion):
        res = {}
        for fusion_name in fusion:
            effect_times = int(fusion[fusion_name][EFFECT_TIMES_KEY])
            if effect_times > 0:
                res[fusion_name] = effect_times
        return res


class Fusion(ToolObject):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion_result = []
        self.log = util.get_log()

    def prepare(self, json_path="./"):
        """Prepare fusion rule manager
        :param json_path: path to fusion_result.json
        :return: None
        """
        util.create_dir(cfg.FUSION_DIR)
        file_path = os.path.join(json_path, FUSION_RESULT_FILE_NAME)
        file_path_local = os.path.join(cfg.FUSION_DIR, FUSION_RESULT_FILE_NAME)
        if not os.path.isfile(file_path):
            if not os.path.isfile(file_path_local):
                self.log.warning("Can not find fusion result json.")
                return
        else:
            shutil.copy(file_path, cfg.FUSION_DIR)
        fe_jsons = self._get_result_jsons(file_path_local)
        for fe_json in fe_jsons:
            self.fusion_result.append(FusionResult(fe_json))

    def check(self):
        """Check fusion rules
        :return: None
        """
        self.log.info("Check effect fusion rule list.")
        for fusion in self.fusion_result:
            graph_fusion_table = self._build_table(
                "Graph Fusion [GraphID: %s]" % fusion.graph_id(),
                fusion.get_effect_graph_fusion(),
            )
            ub_fusion_table = self._build_table(
                "UB Fusion [GraphID: %s]" % fusion.graph_id(),
                fusion.get_effect_ub_fusion(),
            )
            util.print_panel(
                util.create_columns([graph_fusion_table, ub_fusion_table]),
                title="GraphID:" + fusion.graph_id(),
                fit=True,
            )

    @staticmethod
    def _get_result_jsons(file_name):
        result_jsons = []
        with open(file_name, "r") as f:
            txt = f.read()
            sk = []
            start = -1
            for i in range(len(txt)):
                if txt[i] == "{":
                    sk.append("{")
                if txt[i] == "}":
                    sk.pop()
                if len(sk) == 0:
                    result_jsons.append(json.loads(txt[start + 1 : i + 1]))
                    start = i
        return result_jsons

    @staticmethod
    def _build_table(title, fusion):
        table = util.create_table(title, ["Fusion Name", "Effect times"])
        for f in fusion:
            table.add_row(f, str(fusion[f]))
        return table
