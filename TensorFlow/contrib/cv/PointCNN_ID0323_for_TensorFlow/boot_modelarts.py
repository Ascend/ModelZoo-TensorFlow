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

from npu_bridge.npu_init import *
import os
import numpy as np
import argparse
from help_modelarts import obs_data2modelarts,modelarts2obs
import moxing as mox

print(os.system('env'))

if __name__ == '__main__':
    ## Note: the code dir is not the same as work dir on ModelArts Platform!!!
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    # os.makedirs("/cache/dataset/profiling")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
    config = parser.parse_args()

    print("--------config----------")
    for k in list(vars(config).keys()):
        print("key:{}: value:{}".format(k, vars(config)[k]))
    print("--------config----------")

    # os.environ['DUMP_GE_GRAPH'] = '2'
    # os.environ['PRINT_MODEL'] = '1'
    # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
    # os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = "0"

    ## copy dataset from obs to modelarts
    obs_data2modelarts(config)

    # os.makedirs("/cache/result/models/cls/profiling")

    ## start to train on Modelarts platform
    if not os.path.exists(config.modelarts_result_dir):
        os.makedirs(config.modelarts_result_dir)
    # mnist
    command = 'python3.7 %s/train_val_cls.py -t "/cache/dataset/train_files.txt" -v "/cache/dataset/test_files.txt" -s "/cache/result/models/cls/" -m pointcnn_cls -x mnist_x2_l4 -p NPU' % (
        code_dir)

    print("command:", command)
    os.system('pip install -r %s/requirements.txt' % (code_dir))
    os.system(command)