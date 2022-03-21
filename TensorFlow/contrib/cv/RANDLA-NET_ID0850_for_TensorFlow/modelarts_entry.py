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
import os
import argparse
import sys


# 配置所需的动态库和编译自定义算子
os.environ['LD_PRELOAD'] = '/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so'
os.system('bash /home/ma-user/modelarts/user-job-dir/code/compile_op.sh')
# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="s3://randla-net/data_retry/")
parser.add_argument("--train_url", type=str, default="s3:///randla-net/train_out_npu/")
config = parser.parse_args()

print("[CANN-ZhongZhi] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]

print("[CANN-ZhongZhi] work_dir path is [%s]" % (os.getcwd()))
work_dir = os.getcwd()

print("[CANN-ZhongZhi] start run train shell")
# 执行训练脚本
shell_cmd = ("bash %s/npu_train.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))
os.system(shell_cmd)
print("[CANN-ZhongZhi] finish run train shell")




