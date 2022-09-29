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
def _default_device_ip():
    default_device_ip = [
        '192.168.100.100',
        '192.168.101.100',
        '192.168.102.100',
        '192.168.103.100',
        '192.168.100.101',
        '192.168.101.101',
        '192.168.102.101',
        '192.168.103.101',
    ]
    return default_device_ip


def _default_server_id():
    default_server_ip = "10.147.179.27"
    return default_server_ip


def _default_train_pattern(index):
    return ["OneWorker MultiDevice", "MultiWorker MultiDevice"][index]