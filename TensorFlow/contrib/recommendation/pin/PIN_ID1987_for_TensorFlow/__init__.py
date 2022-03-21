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


import socket

config = {}

host = socket.gethostname()
config['host'] = host.lower()
config['location'] = 'apex'

# only data_path is required
if config['host'] == 'altria':
    # config['data_path'] = '/home/kevin/nas/dataset/Ads-RecSys-Datasets'
    config['data_path'] = r'C:\Users\86187\Documents\MyGits\product-nets-distributed\pnn\iPinYou-all'
    config['env'] = 'cpu'
else:
    if config['location'] == 'apex':
        config['data_path'] = '/newNAS/Datasets/MLGroup/Ads-RecSys-Datasets'
    else:
        config['data_path'] = '/home/distributed_train/Data/'
    config['env'] = 'gpu'

