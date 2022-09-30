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
import os

from config_dist import *
from config_utils import _default_train_pattern
from multi_utils import get_default_config, exits_default_config, get_server_info, write_to_json
from config_args import cache_dir


def deal_host_info(server_info, ids, host, env_device_nums):
    
    host_list = host.split(":")
    env_device_nums += int(host_list[1])
    server_id = 'server' + str(ids)
    if len(host_list) == 2 or len(host_list) == 3:
        if len(host_list) == 3:
            print(int(host_list[1]))
            if int(host_list[1]) % 2 == 0:
                device_list = list(map(int, list(host_list[-1])))
                username, password = get_server_info(host_list[0])
                if len(device_list) == int(host_list[1]):
                    server_info[server_id] = {
                        'server_ip': host_list[0],
                        'server_username': username,
                        'server_password': password,
                        'device_list': device_list
                    }
                else:
                    raise ValueError("The information before and after the device dose not match")

        else:
            if int(host_list[-1]) == 8:

                device_list = [i for i in range(8)]
                username, password = get_server_info(host_list[0])
                server_info[server_id] = {
                    'server_ip': host_list[0],
                    'server_username': username,
                    'server_password': password,
                    'device_list': device_list
                }
            else:
                raise ValueError("When the number of devices is not 8, give the specific device")
    else:
        raise ValueError("Incorrect information")

    return server_info, env_device_nums


def get_train_pattern(args):
    device_nums = int(args.np)
    if ',' in args.env:
        server_info = {}
        hosts_list = args.env.split(',')
        if len(hosts_list) % 2 != 0:
            raise ValueError("Multi worker must be a multiple of 2")
        else:
            env_device_nums = 0
            for ids, host in enumerate(hosts_list):
                server_info, env_device_nums = deal_host_info(server_info, ids, host, env_device_nums)

        if device_nums == env_device_nums:
            return _default_train_pattern(1), server_info
    else:
        server_info = {}
        env_device_nums = 0
        server_info, env_device_nums = deal_host_info(server_info, 0, args.env, env_device_nums)
        if device_nums == env_device_nums:
            return _default_train_pattern(0), server_info
        else:
            raise ValueError("device nums != len(device_list)")


def analyze_user_input(args):
    """

    :param args:
    "return"
    """
    train_pattern, env_info = get_train_pattern(args)
    if train_pattern == "OneWorker OneDevice":
        make_onedevice_config(args, env_info)
    elif train_pattern == "OneWorker MultiDevice":
        make_multidevice_config(args, env_info)
    elif train_pattern == "MultiWorker MultiDevice":
        make_multiworker_multidevice_config(args, env_info)
    elif train_pattern == "Host2Host":
        make_h2h_config(args, env_info)
    else:
        pass


def save_server_info(args):
    config = args.config
    if ',' in config:
        server_list = config.split(',')
    else:
        server_list = [config]

    server_dict = dict()
    print(server_dict)
    for server in server_list:
        server_info = server.split(':')
        if server_info[0] not in server_dict:
            server_dict[server_info[0]] = {}
        server_dict[server_info[0]]["server_username"] = server_info[1]
        server_dict[server_info[0]]["server_password"] = server_info[2]
    file_name = 'env_config.json'
    write_to_json(file_name, server_dict)


def put_process_monitor():
    filepath = os.path.dirname(__file__)
    filename = filepath + '/process_monitor.py'
    os.system(f"cp {filename} {cache_dir}")


def config_command(args):
    put_process_monitor()
    analyze_user_input(args)