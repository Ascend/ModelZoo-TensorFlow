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
import os
import numpy as np
import argparse
from help_modelarts import obs_data2modelarts, modelarts_result2obs
import sys
import moxing as mox

if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!

    print("*" * 10, sys.argv[0], "*" * 10)

    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("code_dir: ", code_dir)
    print("work_dir: ", work_dir)

    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--pretrained_url", type=str, default="obs://pfe-npu/pretrained")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    ## copy dataset from obs to modelarts
    obs_data2modelarts(config)
    ## copy pretrained model from obs to modelarts
    mox.file.copy_parallel(config.pretrained_url, config.modelarts_data_dir)
    ## start to train on Modelarts platform
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)
    ##training
    # bash_header = os.path.join(code_dir, 'scripts/run_1p.sh')
    bash_header = os.path.join(code_dir, 'train_testcase.sh')
    print("bash_header:", bash_header)
    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
    print("arg_url:", arg_url)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print(code_dir)
    print("bash command:", bash_command)
    os.system(bash_command)
