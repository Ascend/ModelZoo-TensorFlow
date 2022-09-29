#
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
#
# -*- coding: utf-8 -*-
import argparse


def define_ascend_distribute_flags():
    parser = argparse.ArgumentParser("""NPU Distribute run commond

    ##################################################################################################################################################################

    WARNING: Users only need to focus on '--np', '--env', '--train_command', '--aoe' and '--use_library' parameters, do not change other parameters!
    WARNING: Before using this tool, please ensure you can perform one-device training with the Ascend NPU, and ensure using this tool with the same training command!
    WARNING: Before using this tool, users need to define a config file. For more details, please see "README.md".

    Users can use this tool easily with the follow examples:
    common command format: python3 distribute_npu.py --np (total device mun) --env (ip):(device num):(device id) --train_command "onedevice training command"
    for one-worker-multi-devices training: python3 distribute_npu.py --np 4 --env 10.10.10.10:4:0123 --train_command "bash train.sh"
    for multi-workers-multi-devices training: python3 distribute_npu.py --np 8 --env 10.10.10.10:4:0123,10.10.10.11:4:0123 --train_command "bash train.sh"
    for using AOE tuning tool: python3 distribute_npu.py --np 4 --env 10.10.10.10:4:0123 --train_command "bash train.sh" --aoe=True
    for disable the AOE tuned bank file: python3 distribute_npu.py --np 4 --env 10.10.10.10:4:0123 --train_command "bash train.sh" --use_library=False

    ATTENTION: 1. After successful one-worker-multi-devices training, users can train with multi-workers, just need to modify the '--env' parameter.
               2. When setting the '--env', please using ',' to separate different workers, and do not forget to modify the config file which includes env info.
               3. After successful one-worker-multi-devices training, users can tune with the AOE tool, just need to add '--aoe=True' after the previous command.
               4. After AOE, if a 'xx_gradient_fusion.json' file generated in '/root/ascend_tools/ascend_distribute/custom_tune_bank/' directory, AOE is successful.
               5. Using AOE tuned file is default, users can set '--use_library=False' to disable using AOE tuned file.
               
    ##################################################################################################################################################################        
    """)
    parser.add_argument(
        "--config",
        default=None,
        help="Enter containing server ip:username:password.",
    )

    parser.add_argument(
        "--np",
        default=8,
        type=int,
        help="Necessary, the total number of devices used for training.",
    )

    parser.add_argument(
        "--env",
        default=None,
        help="Necessary, environment information, please input with '--env {ip}:{device num}:{device ip}' format, "
             "when training with MultiWorker, please use ',' to separate different workers",
    )

    parser.add_argument(
        "--train_command",
        default=None,
        type=str,
        help="Necessary, training command, input like --train_command 'bash train_1p.sh' or "
             "--train_command 'python3 train.py'",
    )

    parser.add_argument(
        "--aoe",
        default=False,
        type=bool,
        help="Optional, if or not use AOE, default is False, use --aoe=True to enable",
    )

    parser.add_argument(
        "--use_library",
        default=False,
        type=bool,
        help="Optional, if or not training with custom tune bank file witch generated by AOE, default is False, "
             "use --use_library=True to enable",
    )

    parser.add_argument(
        "--config_file",
        default=None,
    )

    parser.add_argument(
        "--train_log_dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--rank_nums",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--start_rank_id",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--multi_worker",
        action="store_true",
    )

    parser.add_argument(
        "--rank_size",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--worker_num",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--use_config",
        action="store_true",
    )

    parser.add_argument(
        "--command_list",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--server_list",
        type=None,
    )

    return parser