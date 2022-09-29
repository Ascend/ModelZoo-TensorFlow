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
import re
import json

from config_utils import _default_device_ip, _default_server_id
from config_args import BaseConfig, MultiDeviceConfig, MultiWorkerConfig
from multi_utils import get_worker_device_ip, create_rank_table, check_multi_communication, save_rank_table
from multi_utils import save_default_config


def make_onedevice_config(args, env_info):
    pass


def make_multidevice_config(args, env_dict):
    if isinstance(env_dict, dict):
        pass
    else:
        raise ValueError("")
    device_list = env_dict['server0']['device_list']
    device_ip = _default_device_ip()
    server_id = _default_server_id()

    server_info = {
        'device_ip': device_ip,
        'server_id': server_id,
        'device_list': device_list
    }

    create_rank_table(server_info)
    server_list = []
    env_dict['server0']['device_ip'] = None
    env_dict['server0']['rank_id'] = None
    args.rank_size = len(device_list)
    server_list.append(env_dict['server0'])
    save_default_config(len(device_list), server_list)
    args.server_list = server_list
    args.worker_num = 1


def make_multiworker_multidevice_config(args, env_dict):
    if isinstance(env_dict, dict):
        pass
    else:
        pass

    worker_num = len(env_dict)
    server_list = []
    device_ids = []
    for i in range(0, worker_num):
        server_id = 'server' + str(i)
        device_ids.append(len(env_dict[server_id]['device_list']))
        server_list.append(env_dict[server_id])

    rank_id_list = [0]
    for device_id in device_ids[:-1]:
        rank_id_list.append(rank_id_list[-1] + device_id)
    rank_size = sum(device_ids)

    for i in range(worker_num):
        device_ip = get_worker_device_ip(server_list[i])
        server_list[i]['device_ip'] = device_ip
        server_list[i]['rank_id'] = rank_id_list[i]

    check_multi_communication(server_list)
    save_default_config(rank_size, server_list)
    save_rank_table(rank_size, server_list)

    args.rank_size = rank_size
    args.multi_worker = True
    args.worker_num = worker_num
    args.server_list = server_list


def make_h2h_config(args, env_info):
    pass