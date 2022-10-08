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
import signal


def get_parent_process():
    result = os.popen("pgrep -f RANK_TABLE_FILE")
    result = result.read()
    parent_process = []
    for line in result.splitlines():
        parent_process.append([int(line)])
    return parent_process


def get_all_process():
    all_process = os.popen("ps -ef")
    all_process = all_process.read()
    sub_parent = []
    for process in all_process.splitlines():
        process_info = list(filter(lambda x:x, process.split(' ')))

        try:
            sub_parent.append([int(process_info[1]), int(process_info[2])])
        except ValueError as e:
            continue

    sub_parent_dict = {}
    for p in sub_parent:
        if p[1] in sub_parent_dict:
            sub_parent_dict[p[1]].append(p[0])
        else:
            sub_parent_dict[p[1]] = [p[0]]

    return sub_parent_dict


def find_sub_process(process, sub_process, sub_parent_dict):
    temp = []
    for p_pid in process:
        sub_process.add(p_pid)
        if p_pid in sub_parent_dict:
            for s_pid in sub_parent_dict[p_pid]:
                sub_process.add(s_pid)
                temp.append(s_pid)
        else:
            return
    find_sub_process(temp, sub_process, sub_parent_dict)


def find_all_process(parent_process, sub_parent_dict):
    all_sub_parent = []
    for process in parent_process:
        sub_process = set()
        find_sub_process(process, sub_process, sub_parent_dict)
        all_sub_parent.append(sub_process)

    return all_sub_parent


def kill_all_process(all_sub_parent):
    for sub_parent in all_sub_parent:
        for process in sub_parent:
            os.kill(process, signal.SIGKILL)


def main():
    parent_process = get_parent_process()
    sub_parent_dict = get_all_process()
    all_sub_parent = find_all_process(parent_process, sub_parent_dict)
    kill_all_process(all_sub_parent)


if __name__ == '__main__':
    main()