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

import numpy as np

acc = 0.0
input_num = 10000
cell_type = 'lstm'
correct = 0.0

prefix = './data/om_310_output/tf_gru_310/'

# read ground-truth labels
labels = np.load("./data/gt/labels.npy", allow_pickle=True)

# calculate accuracy
for idx in range(input_num):
    pred_path = prefix + '{0:05d}_output_0.bin'.format(idx)
    pred = np.fromfile(pred_path, dtype=np.float16)
    gt = labels[idx]

    #print(pred)
    #print(gt)
    # argmax to get the max value's index of pred and gt
    idx_p, idx_g = np.argmax(pred), np.argmax(gt)
    #print(idx_p)
    #print(idx_g)
    # judge if prediction match the ground truth label
    if idx_p == idx_g:
        correct += 1.0

# final accuracy
acc = correct / input_num * 100

# print final evaluation accuracy
print("======= Final Eval Accuracy =======")
print("Current Environment: Ascend310")
print("Current Om Output Dir Prefix: %s" % prefix)
print("Om Model Cell Type: %s " % cell_type)
print("MNIST Test Set Evaluation Accuracy: %.2f%%" % acc)
