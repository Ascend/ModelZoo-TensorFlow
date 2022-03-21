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
import numpy as np
import argparse
from help_modelarts import obs_data2modelarts_data, obs_data2modelarts_result

print(os.system('env'))

if __name__ == '__main__':
    # Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)  # 40g
    work_dir = os.getcwd()  # /cache NPU 3T
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="/srdrm/result/", help='OBS上的输出路径')
    parser.add_argument("--data_url", type=str, default="/srdrm/dataset/", help='OBS上的数据集路径')
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset", help='在Modelarts容器上的数据集存放路径')
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result", help='在Modelarts容器上创建训练输出目录')
    parser.add_argument("--bash_file_train", type=str, default="train_GANs_npu_1p.sh", help='要运行的训练脚本')
    parser.add_argument("--bash_file_test", type=str, default="test_SR_npu_1p.sh", help='要运行的测试脚本')
    parser.add_argument("--bash_file_measure", type=str, default="measure_gpu_1p.sh", help='要运行的测试脚本')
    # parser.add_argument("--obs_checkpoint_dir", type=str, default="obs://srdrm/modelatrs_result/",
    # help='OBS上的以往训练数据文件')
    parser.add_argument("--obs_checkpoint_dir", type=str, default="obs://srdrm/modelarts_result_mixpre/", help='OBS'
                                                                                                               '上的以往训练数据文件')
    parser.add_argument("--obs_dir", type=str, default="obs://srdrm/", help='OBS上路径')
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    # copy dataset from obs to modelarts
    obs_data2modelarts_data(config)

    # start to train_gpu on Modelarts platform
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)

    # copy last result from obs to modelarts
    obs_data2modelarts_result(config)

    # 执行训练的命令
    bash_header = os.path.join(code_dir, 'scripts/npu_scripts/{}'.format(config.bash_file_train))
    arg_url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.obs_dir)
    bash_command = 'bash %s %s' % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)

