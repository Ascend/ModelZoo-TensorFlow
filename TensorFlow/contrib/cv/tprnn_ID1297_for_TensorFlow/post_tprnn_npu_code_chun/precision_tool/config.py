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

# Dump config '0|5|10'
TF_DUMP_STEP = "0"

# path to run package operator cmp compare
# default may be /usr/local/Ascend/
CMD_ROOT_PATH = "/usr/local/Ascend/"

# ASCEND Log Path
ASCEND_LOG_PATH = "/root/ascend/log/plog/"

# TOOL CONFIG
LOG_LEVEL = "NOTSET"
ROOT_DIR = "/cache/overflow_data"

# [train/infer] if adapt from msquickcmp result, set net type to infer
NET_TYPE = "train"

"""
precision_data/
├── npu
│   ├── debug_0
|   |   ├── dump
|   |   |   └── 20210510101133
|   │   └── graph
|   |       └── ge_proto_00000179_PreRunAfterBuild.txt
│   └── debug_1
├── tf
|   ├── tf_debug
|   └── dump
├── overflow
├── fusion
└── temp
    ├── op_graph
    ├── decode
    |   ├── dump_decode
    |   ├── overflow_decode
    |   └── dump_convert
    └── vector_compare
        ├── 20210510101133
        |   ├── result_123456.csv
        |   └── result_123455.csv
        └── 20210510101134
            └── result_123458.csv
"""

# Static dirs, do not change
DATA_ROOT_DIR = os.path.join(ROOT_DIR, "precision_data")

# fusion
FUSION_DIR = os.path.join(DATA_ROOT_DIR, "fusion")

# npu dump/graph parent dir
NPU_DIR = os.path.join(DATA_ROOT_DIR, "npu")
DEFAULT_NPU_DIR = os.path.join(NPU_DIR, "debug_0")
DEFAULT_NPU_DUMP_DIR = os.path.join(DEFAULT_NPU_DIR, "dump")
DEFAULT_NPU_GRAPH_DIR = os.path.join(DEFAULT_NPU_DIR, "graph")
DEFAULT_OP_DEBUG_DIR = DEFAULT_NPU_DIR

# npu overflow dir
OVERFLOW_DIR = os.path.join(DATA_ROOT_DIR, "overflow")
NPU_OVERFLOW_DUMP_DIR = os.path.join(OVERFLOW_DIR, "dump")

# tf dirs
TF_DIR = os.path.join(DATA_ROOT_DIR, "tf")
TF_DEBUG_DUMP_DIR = os.path.join(TF_DIR, "tf_debug")
TF_DUMP_DIR = os.path.join(TF_DIR, "dump")
TF_GRAPH_DIR = os.path.join(TF_DIR, "graph")

# tmp dirs
TMP_DIR = os.path.join(DATA_ROOT_DIR, "temp")
OP_GRAPH_DIR = os.path.join(TMP_DIR, "op_graph")

DECODE_DIR = os.path.join(TMP_DIR, "decode")
OVERFLOW_DECODE_DIR = os.path.join(DECODE_DIR, "overflow_decode")
DUMP_DECODE_DIR = os.path.join(DECODE_DIR, "dump_decode")
DUMP_CONVERT_DIR = os.path.join(DECODE_DIR, "dump_convert")

VECTOR_COMPARE_PATH = os.path.join(TMP_DIR, "vector_compare")
TF_TENSOR_NAMES = os.path.join(TMP_DIR, "tf_tensor_names.txt")
TF_TENSOR_DUMP_CMD = os.path.join(TMP_DIR, "tf_tensor_cmd.txt")

# FLAG
PRECISION_TOOL_OVERFLOW_FLAG = "PRECISION_TOOL_OVERFLOW"
PRECISION_TOOL_DUMP_FLAG = "PRECISION_TOOL_DUMP"

# DUMP CONFIG
OP_DEBUG_LEVEL = 0
DUMP_GE_GRAPH_VALUE = 2
DUMP_GRAPH_LEVEL_VALUE = 3
DUMP_SEED = 0

# TF_DEBUG
TF_DEBUG_TIMEOUT = 360

# MSACCUCMP
MS_ACCU_CMP = r"msaccucmp.py[c]?"
BUILD_JSON_GRAPH_NAME = "Build"
