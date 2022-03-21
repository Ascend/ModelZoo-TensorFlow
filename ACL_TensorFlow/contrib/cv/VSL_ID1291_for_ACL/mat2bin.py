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


import h5py
import numpy as np

# define network structure, parameters
global_latent_dim  = 10
local_latent_dim   = 5
local_latent_num   = 5
obj_res     = 30
batch_size  = 100

# load dataset (pick modelnet40 or modelnet10)
data = h5py.File('dataset/ModelNet10_res30_raw.mat')
train_all = np.transpose(data['train'])
test_all = np.transpose(data['test'])

# select test_all to make the .bin data
# np.random.shuffle(test_all)
i = 5
train = train_all[batch_size * i:batch_size * (i + 1), 1:].reshape(
    [batch_size, obj_res, obj_res, obj_res, 1])
train.tofile("dataset/bin/ModelNet10.bin")
