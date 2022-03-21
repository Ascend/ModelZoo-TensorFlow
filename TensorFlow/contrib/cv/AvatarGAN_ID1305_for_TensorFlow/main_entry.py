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
import moxing as mox

parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/cache/datasets")
parser.add_argument("--train_url", type=str, default="/cache/output")
config = parser.parse_args()

os.environ['LD_PRELOAD'] = '/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so'
exec_cmd = ("python3.7 /home/ma-user/modelarts/user-job-dir/code/main.py --data_url %s --train_url %s"
            % (config.data_url, config.train_url))
os.system(exec_cmd)



