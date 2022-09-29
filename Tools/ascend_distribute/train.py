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
import argparse
import sys
import signal
from multiprocessing import Process

from config_args import cache_dir, default_config_file, load_config_from_file, library_path, bank_path
from multi_utils import ssh_server
from process_manage import ProcessManage


def create_train_command(args, ids, device_id, i=0, ssh=None, exec_path=None):
    if not ssh:
        os.environ['RANK_ID'] = f'{ids + args.start_rank_id}'
        os.environ['ASCEND_DEVICE_ID'] = f'{device_id}'

    return args.train_command


def run_train_command(args, ids, device, i=0, ssh=None, exec_path=None):
    train_command = create_train_command(args, ids, device, i=i, ssh=ssh)
    print(os.getpid(), os.getppid())
    os.system(train_command)


def run_multi_command(args, ids, device, worker_info, i=0, exec_path=None):
    ssh = ssh_server(worker_info)
    train_command = create_train_command(args, ids, device, i=i, ssh=ssh, exec_path=exec_path)
    rank_table_file = 'rank_table_' + str(args.worker_num) + '_' + str(args.rank_size) + 'p.json'
    source_env1 = 'source /usr/local/Ascend/bin/setenv.bash'
    source_env2 = 'source /usr/local/Ascend/latest/bin/setenv.bash'
    source_env3 = 'source ~/.bashrc'
    RANK_TABLE_FILE = cache_dir + rank_table_file
    RANK_ID = ids + args.server_list[i]['rank_id']
    rank_table_command = f"export RANK_TABLE_FILE={RANK_TABLE_FILE}"
    rank_id_command = f"export RANK_ID={RANK_ID}"
    rank_size_command = f"export RANK_SIZE={args.rank_size}"
    device_id_command = f"export ASCEND_DEVICE_ID={device}"
    get_path = f"cd {exec_path}"
    if args.use_library:
        library_command = f"export TUNE_BANK_PATH={library_path}"
        ssh_in, ssh_out, ssh_error = ssh.exec_command(f"{rank_table_command};{rank_id_command};{device_id_command};{library_command};"
                                                      f"{rank_size_command};{get_path};{source_env1};{source_env2};{source_env3};{train_command}")
    else:
        ssh_in, ssh_out, ssh_error = ssh.exec_command(f"{rank_table_command};{rank_id_command};{device_id_command};"
                                                      f"{rank_size_command};{get_path};{source_env1};{source_env2};{source_env3};{train_command}")
    result = ssh_out.read().decode() or ssh_error.read().decode()
    print("result=", result)
    ssh.close()


def npu_distribute_run(args, process_list):
    if args.rank_nums != len(args.server_list[0]['device_list']):
        print("rank_nums != len(device_list), use len(device_list)!")
    os.environ['RANK_SIZE'] = f'{args.rank_size}'
    rank_table_file = 'rank_table_' + str(args.worker_num) + '_' + str(args.rank_size) + 'p.json'
    os.environ['RANK_TABLE_FILE'] = cache_dir + rank_table_file

    if args.aoe:
        os.environ['AOE_MODE'] = '4'
        if not os.path.exists(library_path):
            os.mkdir(library_path)
        os.environ['TUNE_BANK_PATH'] = library_path
        p = Process(target=run_train_command, args=(args, 0, 0))
        p.start()
        p.join()

    else:
        if args.use_library:
            os.environ['TUNE_BANK_PATH'] = library_path
        for ids, device in enumerate(args.server_list[0]['device_list']):
            p = Process(target=run_train_command, args=(args, ids, device))
            p.start()
            process_list.append(p)


def aoe_check():
    if not os.popen('lspci').readlines():
        raise ValueError("no lspci command")

    if not os.popen('aoe').readlines():
        raise ValueError("no aoe command")


def npu_multi_worker_run(i, args, exec_path, process_list):
    worker_info = args.server_list[i]
    for ids, device in enumerate(worker_info['device_list']):
        p = Process(target=run_multi_command, args=(args, ids, device, worker_info, i, exec_path))
        p.start()
        process_list.append(p)


def run_command(args):
    pm = ProcessManage(args)
    signal.signal(signal.SIGINT, pm.after_treatment)
    if not args.train_command:
        raise ValueError("'--train_command' is must")

    if args.aoe is True and args.use_library is True:
        raise ValueError("cannot apply '--aoe' and '--use_library' at the same time!")

    if args.aoe:
        aoe_check()

    if args.use_library and not os.path.exists(bank_path):
        raise ValueError("no custom tune bank file, please use '--aoe=True' to generate custom tune bank")

    if args.worker_num > 1:
        exec_path = os.getcwd()
        process_list = []
        for i in range(args.worker_num):
            if i == 0:
                npu_distribute_run(args, process_list)
            else:
                npu_multi_worker_run(i, args, exec_path, process_list)

        monitor_process = Process(target=pm.host_process_monitor)
        monitor_process.start()
        process_list.append(monitor_process)
        for p in process_list:
            p.join()
    else:
        process_list=[]
        npu_distribute_run(args, process_list)
        for p in process_list:
            p.join()