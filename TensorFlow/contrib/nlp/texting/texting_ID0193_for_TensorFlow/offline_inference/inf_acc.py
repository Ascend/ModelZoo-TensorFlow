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

lbl = np.fromfile('bin_data/labels/a.bin', dtype=np.float32)

inf = np.array([], dtype=np.float32)
with open('PATH/texting_base_output_0.txt', 'r') as f:
  lines = f.readlines()
for line in lines:
  if line != '\n':
    inf = np.append(inf, int(line.split()[0]))

lbl_resh = np.reshape(lbl, [-1, 2])
correct_prediction = np.equal(inf, np.argmax(lbl_resh, 1))
accuracy_all = correct_prediction.astype(np.float32)
accuracy = np.mean(accuracy_all)

print('accuracy:', accuracy)
