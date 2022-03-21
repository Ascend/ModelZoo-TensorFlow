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
from help_modelarts import obs_data2modelarts

print(os.system('env'))

if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")  #obs中用来存储训练结果的路径,对应配置中的obs path
    parser.add_argument("--data_url", type=str, default="./dataset")  #obs中用来存储训练数据的路径，对应配置中的data path in obs
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/hazy-clear-image-dataset")  #modelarts中用来存储训练数据的路径
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")  #modelarts中用来存储训练结果的路径
    parser.add_argument("--modelarts_logs_dir", type=str, default="/cache/result/logs")
    parser.add_argument("--modelarts_checkpoints_dir", type=str, default="/cache/result/checkpoints")
    parser.add_argument("--modelarts_models_dir", type=str, default="/cache/result/models")
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
    if not os.path.exists(config.modelarts_logs_dir):
        os.makedirs(config.modelarts_logs_dir)
    if not os.path.exists(config.modelarts_checkpoints_dir):
        os.makedirs(config.modelarts_checkpoints_dir)
    if not os.path.exists(config.modelarts_models_dir):
        os.makedirs(config.modelarts_models_dir)
    bash_header = os.path.join(code_dir, 'train_npu.sh')
    arg_url = '%s %s %s %s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir,
                                        config.modelarts_logs_dir, config.modelarts_checkpoints_dir,
                                        config.modelarts_models_dir, config.train_url)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)