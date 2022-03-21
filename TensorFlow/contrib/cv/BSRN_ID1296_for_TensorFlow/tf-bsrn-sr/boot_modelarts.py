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
    parser.add_argument("--train_url", type=str, default="./output", help='OBS上的输出路径')
    parser.add_argument("--data_url", type=str, default="./dataset", help='OBS上的数据集路径')
    #parser.add_argument("--modelarts_data_dir", type=str, default="/cache/bsrn-dataset/DIV2K",help='在Modelarts容器上的数据集存放路径')
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache",
                        help='在Modelarts容器上的数据集存放路径')
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result", help='在Modelarts容器上创建训练输出目录')
    parser.add_argument("--obs_dir", type=str, default="obs://bsrn-test/bsrn", help='OBS上路径')
    # parser.add_argument("--dataloader", type=str, default="div2k_loader",help='数据获取器')
    # parser.add_argument("--data_input_path", type=str, default="/cache/bsrn-dataset/DIV2K/DIV2K_train_LR_bicubic",
    #                     help='在Modelarts容器上的输入数据集存放路径')
    # parser.add_argument("--data_truth_path", type=str, default="/cache/bsrn-dataset/DIV2K/DIV2K_train_HR",
    #                     help='在Modelarts容器上的数据集标签存放路径')
    # parser.add_argument("--model", type=str, default="bsrn", help='模型名称')
    # parser.add_argument("--scales", type=str, default="2,3,4", help='模型降采样系数')
    # parser.add_argument("--bsrn_clip_norm", type=int, default=5, help='Clipping ratio of gradient clipping')
    # parser.add_argument("--num_gpus", type=int, default=1)
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

    #bash_header = os.path.join(code_dir, 'scripts/run_gpu.sh')
    # bash_header = os.path.join(code_dir, 'scripts/run_npu.sh')
    # bash_header = os.path.join(code_dir, 'scripts/validate_npu.sh')
    bash_header = os.path.join(code_dir, 'scripts/run_npu_restore.sh')
    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.obs_dir)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)
