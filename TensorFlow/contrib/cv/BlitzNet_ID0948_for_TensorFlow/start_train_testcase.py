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
"""
This is the boot file for ModelArts platform.
Firstly, the train datasets are copyed from obs to ModelArts.
Then, the string of train shell command is concated and using 'os.system()' to execute
"""
from npu_bridge.npu_init import *
import os
import numpy as np
import argparse
from help_modelarts import obs_data2modelarts

print(os.system('env'))

if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    #Edit Training Job Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")    #OBS Path
    parser.add_argument("--data_url", type=str, default="./dataset")    #Data Path in OBS
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/bliznet-dataset")     #Data Path in modelArts
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")    #result Path in modelArts
    #parser.add_argument("--num_gpus", type=int, default=1)
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    ## copy dataset from obs to modelarts
    obs_data2modelarts(config)

    ## start to train on Modelarts platform
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)
    bash_header = os.path.join(code_dir, 'scripts/train_testcase.sh') #train_testcase.sh
    arg_url = '%s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)
