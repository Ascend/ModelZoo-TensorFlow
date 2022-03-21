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
# Copyright 2020 Huawei Technologies Co., Ltd
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

inf = np.array([], dtype=np.int32)
with open('PATH/gat_base_output_0.txt', 'r') as f:
  lines = f.readlines()
for line in lines:
  if line != '\n':
    inf = np.append(inf, int(line.split()[0]))

lbl = np.fromfile('bin_data/lbl_in/a.bin', dtype=np.int32)
msk = np.fromfile('bin_data/msk_in/a.bin', dtype=np.int32)

lbl_resh = np.reshape(lbl, [-1, 7])
correct_prediction = np.equal(inf, np.argmax(lbl_resh, 1))
accuracy_all = correct_prediction.astype(np.float32)
mask = msk.astype(dtype=np.float32)
mask /= np.mean(mask)
accuracy_all *= mask
accuracy = np.mean(accuracy_all)

print('accuracy:', accuracy)
