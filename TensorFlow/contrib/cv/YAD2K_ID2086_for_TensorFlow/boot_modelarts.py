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
    print("+++++++++++++++++++++++++++code_dir++++++++++++++++++++")
    print("code_dir", code_dir)  # code_dir /home/work/user-job-dir/code
    work_dir = os.getcwd()
    print("+++++++++++++++++++++++++++work_dir++++++++++++++++++++")
    print(work_dir)  # /cache/user-job-dir/workspace/device0
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    # bos the first
    parser.add_argument("--train_url", type=str, default="./output")  # s3://yolov2/yolov2forfen/output/V0003/
    parser.add_argument("--data_url", type=str, default="./dataset")  # s3://yolov2/DataSet/

    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/DataSet")  # /cache/DataSet
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")  # /cache/result

    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    ## copy dataset from obs to modelarts  Need help_moderarts file
    obs_data2modelarts(config)

    ## start to train on Modelarts platform
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)

    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
    print("++++++++++++++++++print the four words++++")
    print("code_dir =    ", code_dir)
    print("config.modelarts_data_dir =    ", config.modelarts_data_dir)
    print("config.modelarts_result_dir =    ", config.modelarts_result_dir)
    print("config.train_url= ", config.train_url)

    # # Use for Train
    # bash_header = os.path.join(code_dir, 'run_1p.sh')  # Train sh
    # bash_command = 'bash %s %s' % (bash_header, arg_url)
    # print("bash command:", bash_command)
    # os.system(bash_command)

    # # Use for Detect AND generate the AP dir
    # detect_bash = os.path.join(code_dir,'detect_1p.sh') # detect sh
    # detect_command = 'bash %s %s' % (detect_bash, arg_url)
    # print("detect command:",detect_command)
    # os.system(detect_command)

    # # Use for Testcase
    self_batch = os.path.join(code_dir, 'train_testcase.sh')  # testcase sh
    self_command = 'bash %s %s' % (self_batch, arg_url)
    print("self_command:", self_command)
    os.system(self_command)
