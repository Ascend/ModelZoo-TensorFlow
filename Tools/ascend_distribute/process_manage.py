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
import sys
import time

from config_args import cache_dir
from multi_utils import ssh_server


class ProcessManage:
    def __init__(self, args):
        self.args = args

    def sub_process_monitor(self):
        for i in range(1, self.args.worker_num):
            worker_info = self.args.server_list[i]
            ssh = ssh_server(worker_info)
            ssh_in, ssh_out, ssh_error = ssh.exec_command("pgrep -f RANK_TABLE_FILE")
            result = ssh_out.read().decode() or ssh_error.read().decode()
            process = result.splitlines()
            if len(process) == len(worker_info['device_list']):
                print(f"{worker_info['server_ip']} has running {len(process)} Process!")

    def host_process_monitor(self):
        result = os.popen("pgrep -f distribute_npu")
        process = result.read().splitlines()
        if len(process)-2 == self.args.rank_size:
            print(f"Have Run {len(process)-2} processes!")
        else:
            print(f"Only found {len(process)-2} processes, not equal rank_size!")

        i = 0
        while len(process) - 2:
            i += 1
            time.sleep(30)
            result = os.popen("pgrep -f distribute_npu")
            process = result.read().splitlines()
            if i % 20 == 0:
                self.sub_process_monitor()

    def after_treatment(self, signal, frame):
        for i in range(1, self.args.worker_num):
            worker_info = self.args.server_list[i]
            ssh = ssh_server(worker_info)
            ssh_in, ssh_out, ssh_error = ssh.exec_command(f"python3 {cache_dir}/process_monitor.py")
            result = ssh_out.read().decode() or ssh_error.read().decode()
        sys.exit(0)