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
import multiprocessing
import datetime
import time
import argparse
from help_modelarts import obs_data2modelarts, copy_modelarts_result2obs

print(os.system('env'))

def copy_modelarts2obs(config):
    print(datetime.datetime.now())
    time.sleep(5*60)
    obs_result_dir = os.path.join(config.train_url, '../result')
    copy_modelarts_result2obs(config.modelarts_result_dir, obs_result_dir)

if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!可以都不动
    #code_dir = os.path.dirname(__file__)
    code_dir = os.path.realpath("/home/work/user-job-dir/code")
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    #引入外部参数，即对应Edit Training Job Configuration中的输入
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")    #OBS Path
    parser.add_argument("--data_url", type=str, default="./dataset")    #Data Path in OBS
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/coco-dataset")     #Data Path in modelArts，就存在cache下，目录可以新建
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")    #result Path in modelArts
    parser.add_argument("--num_gpus", type=int, default=1)  ##若在modelArts的GPU上进行训练则需要代码
    parser.add_argument("--obs_dir", type=str, default="obs://model-arts-1/spinenet_npu/")
    parser.add_argument("--platform", type=str, default="modelarts")
    parser.add_argument("--result", type=str, default="/cache/result")
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    ## copy dataset from obs to modelarts
    obs_data2modelarts(config)

    # 创建一个进程，args传参 必须是元组
    #p = multiprocessing.Process(target=copy_modelarts2obs, args=(config,))
    #p.start()

    ## start to train on Modelarts platform
    modelarts_profiling_dir = "/tmp/npu_profiling"
    if not os.path.exists(modelarts_profiling_dir):
        os.makedirs(modelarts_profiling_dir)
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)
    bash_header = os.path.join(code_dir, 'tpu/scripts/run_1p.sh') #按需改写run_1p.sh
    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)
