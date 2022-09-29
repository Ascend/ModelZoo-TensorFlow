# -*- coding: utf-8 -*-
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
import os
import time
import json

import paramiko

from config_args import cache_dir, default_rank_table_file, default_config_file, library_path, bank_path


def get_default_hccn_conf():
    hccn_conf = os.path.join("/etc", "hccn.conf")

    if os.path.isfile(hccn_conf) and os.path.getsize(hccn_conf) != 0:
        default_hccn_conf = hccn_conf
    else:
        default_hccn_conf = None
    return default_hccn_conf


def ssh_server(server_info):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname=server_info['server_ip'], port=22,
                    username=server_info['server_username'], password=server_info['server_password'])
        return ssh
    except Exception as e:
        print(e)
        return


def get_worker_device_ip(server_info):
    device_ip = []
    ssh = ssh_server(server_info)
    default_hccn_conf = get_default_hccn_conf()

    if default_hccn_conf:
        ssh_in, ssh_out, ssh_error = ssh.exec_command(f"cat {default_hccn_conf}")
        result = ssh_out.read() or ssh_error.read()
        result = result.decode().strip().split('\n')
        for i in range(8):
            address = 'address_' + str(i)
            for info in result:
                if address in info:
                    device_ip.append(info.split("=")[-1])

        ssh.exec_command(f"mkdir -p {cache_dir}")
        ssh.exec_command(f"mkdir -p {library_path}")
        ssh.close()
        return device_ip
    else:
        print("Not found /etc/hccn.conf, please prepare /etc/hccn.conf.")


def write_to_json(file_name, dict):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    json_dict = json.dumps(dict)
    json_file = os.path.join(cache_dir, file_name)
    f = open(json_file, 'w')
    f.write(json_dict)
    f.close()
    return json_file


def table_dict(server_count, server_list_json):
    return {
        "server_count": str(server_count),
        "server_list": server_list_json,
        "status": "completed",
        "version": "1.0"
    }


def create_rank_table(server_info):
    rank_id_start = 0
    server_count = 1
    rank_size = len(server_info['device_list'])
    device = []
    server_list_json = []

    for dev in server_info['device_list']:
        rank_id = rank_id_start
        rank_id_start += 1
        device.append({"device_id": str(dev), "device_ip": str(server_info['device_ip'][dev]), "rank_id": str(rank_id)})
    server_list_json.append({"server_id": str(server_info['server_id']), "device": device})

    rank_table_dict = table_dict(server_count, server_list_json)
    file_name = 'rank_table_' + str(server_count) + '_' + str(rank_size) + 'p.json'
    write_to_json(file_name, rank_table_dict)


def check_sub_server(server):
    server_0 = server[0]
    server_1 = server[1]

    def check_each(host_server, target_server):
        ssh_host = ssh_server(host_server)
        for i in range(len(target_server['device_ip'])):
            ssh_host.exec_command(f"cat /e hccn_tool -i {i} -netdetect -s address {target_server['device_ip'][i]}")

            time.sleep(5)
            ssh_in, ssh_out, ssh_error = ssh_host.exec_command(f"hccn_tool -i {i} -net_health -g")
            result = ssh_out.read() or ssh_error.read()
            result = result.decode().strip()
            if "Success" in result:
                print(f"{host_server['server_ip']} device {i} ->"
                      f"{target_server['server_ip']} device {i} Success!!!")
        ssh_host.close()

    # A -> B
    check_each(server_0, server_1)
    # B -> A
    check_each(server_1, server_0)


def check_multi_communication(server_list):
    double_server = [(server_list[i], server_list[j])
                     for i in range(len(server_list))
                     for j in range(len(server_list)) if i < j]

    for server in double_server:
        check_sub_server(server)


def sftp_server(server_info):
    transport = paramiko.Transport((server_info['server_ip'], 22))
    try:
        transport.connect(username=server_info['server_username'], password=server_info['server_password'])
    except Exception as e:
        print(e)
        return

    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp


def put_rank_table(sftp, rank_table_file):
    sftp.put(rank_table_file, rank_table_file)


def upload_table_file(server_list, rank_table_file):
    for server in server_list[1:]:
        transport, sftp = sftp_server(server)
        put_rank_table(sftp, rank_table_file)
        transport.close()


def get_server_rank(server):
    server_id = server["server_ip"]
    rank_id_start = server['rank_id']
    device = []
    for dev in server['device_list']:
        rank_id = rank_id_start
        rank_id_start += 1
        device.append({"device_id": str(dev), "device_ip": str(server['device_ip'][dev]), "rank_id": str(rank_id)})

    return {"server_id": str(server_id), "device": device}


def save_rank_table(rank_size, server_list):
    server_count = len(server_list)

    server_list_json = []
    for server in server_list:
        server_list_json.append(get_server_rank(server))

    rank_table_dict = table_dict(server_count, server_list_json)
    file_name = 'rank_table_' + str(server_count) + '_' + str(rank_size) + 'p.json'
    rank_table_file = write_to_json(file_name, rank_table_dict)
    process_file = cache_dir + 'process_monitor.py'
    upload_table_file(server_list, rank_table_file)
    upload_table_file(server_list, process_file)
    if bank_path is not None and os.path.exists(bank_path):
        upload_table_file(server_list, bank_path)


def save_default_config(rank_size, server_list):
    server_count = len(server_list)
    server_dict = {'server_info': server_list}
    file_name = 'default_config_' + str(server_count) + '_' + str(rank_size) + 'p.json'
    write_to_json(file_name, server_dict)


def get_default_config(config_file, multi_worker=True):
    with open(config_file, "r") as config:
        load_dict = json.load(config)
        if multi_worker:
            server_info = {}
            server_list = load_dict['server_info']
            for ids, server in enumerate(server_list):
                server_id = 'server' + str(ids)
                server_info[server_id] = {
                    'server_ip': server['server_ip'],
                    'server_username': server['server_username'],
                    'server_password': server['server_password'],
                    'device_list': server['device_list']
                }
        else:
            server_info = load_dict['server_info'][0]['device_list']
    return server_info


def exits_default_config(config_file):
    return os.path.exists(config_file)


def get_server_info(host_ip):
    config_file = cache_dir + "/env_config.json"
    if exits_default_config(config_file):
        with open(config_file, "r") as config:
            try:
                load_dict = json.load(config)
                return load_dict[host_ip]['server_username'], load_dict[host_ip]['server_password']
            except Exception as e:
                print(e)
                return