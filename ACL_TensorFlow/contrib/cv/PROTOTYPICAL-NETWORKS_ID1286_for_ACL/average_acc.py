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

"""
    Average accuracy calculation
"""

n_test_episodes = 1000

avg_acc = 0.

for epi in range(n_test_episodes):
    filename = "/home/HwHiAiUser/proto_pb/out/output1/2021123_16_44_44_641829/input" + str(epi) + "_output_0.txt"
    with open(filename, 'r') as f:
        acc_test = f.read()
    avg_acc += float(acc_test)

avg_acc /= n_test_episodes
print('Average Test Accuracy: {:.5f}'.format(avg_acc))
