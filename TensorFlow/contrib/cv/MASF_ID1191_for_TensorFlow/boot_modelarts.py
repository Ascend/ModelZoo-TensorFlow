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
# ===========================
#   Author      : ChenZhou
#   Time        : 2021/11
#   Language    : Python
# ===========================
"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copyed from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""
import os
import argparse
import moxing as mox
os.environ['LD_PRELOAD'] = "/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so"
os.system("pip3 install scikit-image")
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="obs://masf/masfdata/")
parser.add_argument("--train_url", type=str, default="obs://masf/masfdata/masflog/")
config = parser.parse_args()
# 在ModelArts容器创建数据存放目录
data_dir = "/home/ma-user/modelarts/user-job-dir/code/"
# OBS数据拷贝到ModelArts容器内
mox.file.copy_parallel("obs://masf/masfdata/", data_dir)
#运行主脚本
os.system('python3.7 /home/ma-user/modelarts/user-job-dir/code/main.py')
#将训练结果拷贝回OBS
mox.file.copy_parallel("/home/ma-user/modelarts/user-job-dir/code/log/", config.train_url)