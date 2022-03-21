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

"""boot py-file, which call the shell script to run the predict_3dpose.py"""

import os, subprocess
import argparse
from help_modelarts import obs_data2modelarts

print(os.system('env'))

if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code.dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./result")
    parser.add_argument("--data_url", type=str, default="./data")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/h36m")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
    config = parser.parse_args()

    print("------------config------------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("------------config------------")

    obs_data2modelarts(config)

    os.mkdir("/cache/profiling")  # for profiling

    if not os.path.exists(config.modelarts_result_dir):
        os.mkdir(config.modelarts_result_dir)
    script_path = "scripts/train_performance.sh"
    authorize_command = "chmod +x %s" % os.path.join(code_dir, script_path)
    print(authorize_command)
    os.system(authorize_command)
    bash_header = os.path.join(code_dir, script_path)
    arg_url = "%s %s %s %s" % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
    # bash_command = 'bash %s %s' % (bash_header, arg_url)
    bash_command = '%s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)

    # p = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE)
    # out = p.stdout.readlines()
    # with open(os.path.join(config.train_url, "command.log"), 'w') as logfile:
    #     for line in out:
    #         logfile.write(line.strip().decode('UTF-8'))
