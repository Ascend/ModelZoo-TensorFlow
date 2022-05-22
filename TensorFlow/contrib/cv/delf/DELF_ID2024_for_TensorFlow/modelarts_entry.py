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

import os
import argparse
import sys

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/cache/dataset")
parser.add_argument("--train_url", type=str, default="/cache/output")
config = parser.parse_args()

print("[CANN-ZhongZhi] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]

print("[CANN-ZhongZhi] work_dir path is [%s]" % (os.getcwd()))
work_dir = os.getcwd()

print("[CANN-ZhongZhi] start run train shell")
## [Training]
shell_cmd = ("bash %s/npu_train.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))

## [Eval]
# shell_cmd = ("bash %s/npu_eval.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))

import os
dump_data_dir = "/cache/dump_data"
os.makedirs(dump_data_dir)
profile_data_dir = "/cache/profile_data"
os.makedirs(profile_data_dir)

os.system(shell_cmd)
print("[CANN-ZhongZhi] finish run train shell")

import moxing as mox
mox.file.copy_parallel(dump_data_dir, config.train_url)
mox.file.copy_parallel(profile_data_dir, config.train_url)