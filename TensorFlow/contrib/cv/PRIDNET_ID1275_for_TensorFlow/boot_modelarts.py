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
import argparse
import moxing as mox


def is_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_url", type=str, default="/cache/pridnet_dataset")
    parser.add_argument("--modelarts_result_url", type=str, default="/cache/result/")
    parser.add_argument("--modelarts_train_model_url", type=str, default="/cache/train_model/")
    # gpu env
    # parser.add_argument("--num_gpus", type=int, default=1)
    config = parser.parse_args()

    # copy dataset from obs to modelarts
    mox.file.copy_parallel(src_url=config.data_url, dst_url=config.modelarts_data_url)
    files = os.listdir(config.modelarts_data_url)
    print("------Flies:", files)

    # start to train on modelarts
    is_exist(config.modelarts_result_url)
    is_exist(config.modelarts_train_model_url)
    bash_header = os.path.join(code_dir, "scripts/run_modelarts.sh")
    arg_url = "%s %s %s %s %s" % (
    code_dir, config.modelarts_data_url, config.modelarts_result_url, config.modelarts_train_model_url,
    config.train_url)
    bash_command = "bash %s %s" % (bash_header, arg_url)
    print("bash command:", bash_command)
    os.system(bash_command)
