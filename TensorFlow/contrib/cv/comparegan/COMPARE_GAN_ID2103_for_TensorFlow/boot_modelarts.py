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
import datetime
import moxing as mox
import numpy as np
import argparse
from help_modelarts import obs_data2modelarts
print(os.system('env'))

if __name__ == '__main__':


    # Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/datasets")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    # copy dataset from obs to modelarts
    #Copy train data from obs to modelarts by using moxing api.

    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(config.data_url, config.modelarts_data_dir))
    mox.file.copy_parallel(src_url=config.data_url, dst_url=config.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(config.modelarts_data_dir)
    print("===>>>Files:", files)

    # start to train on Modelarts platform
    print("===>>>Begin training:")
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)
    bash_header = os.path.join(code_dir, 'compare_gan/run_modelarts_1p.sh')
    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)
    print("===>>>Training finished:")
    #mox.file.copy_parallel(config.modelarts_result_dir, config.train_url)
    # 完成训练后将我们需要保留的中间结果拷贝到obs，目的obs路径为我们之前传入的--train_url
    #  copy results from local to obs
    #local_dir = '/cache/result'
    remote_dir = os.path.join(config.train_url, 'result')
    if not mox.file.exists(remote_dir):
        mox.file.make_dirs(remote_dir)
    start = datetime.datetime.now()
    print("===>>>Copy files from local dir:{} to obs:{}".format(config.modelarts_result_dir, remote_dir))
    mox.file.copy_parallel(src_url=config.modelarts_result_dir, dst_url=remote_dir)
    #AutoTune_Bank
    #print('------------------AutoTune_Bank开始了----------------------')
    #mox.file.copy_parallel("/cache/AutoTune_Bank", 's3://comparegan/auto')
    #print('------------------AutoTune_Bank结束了----------------------')
    end = datetime.datetime.now()
    print("===>>>Copy from local to obs, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(config.modelarts_result_dir)
    print("===>>>Files number:", len(files))
