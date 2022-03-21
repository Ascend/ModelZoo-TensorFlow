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
from sklearn import svm

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

# np.random.shuffle(train_all)
test_batch = 1
test_feature = np.loadtxt('dataset/modelnet10_output_0.txt')

clf = svm.SVC(kernel='rbf')
clf.fit(test_feature[:, :], test_all[0:batch_size * test_batch, 0])

test_accuracy = np.sum(test_all[0:batch_size * test_batch, 0] == clf.predict(test_feature[:, :])) / (
      test_batch * batch_size)

print('Shape classification: test: {:.4f}'
      .format(test_accuracy))