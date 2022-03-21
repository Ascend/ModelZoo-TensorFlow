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

lbl = np.fromfile('bin_data/target/1.bin', dtype=np.float32)
inf = np.fromfile('PATH/1_output_0.bin', dtype=np.float32)

for i in range(2, 119):
    lbl = np.concatenate((lbl, np.fromfile('bin_data/target/{}.bin'.format(i), dtype=np.float32)))
    inf = np.concatenate((inf, np.fromfile('PATH/{}_output_0.bin'.format(i), dtype=np.float32)))

accuracy = np.mean(np.equal(np.round(inf), lbl).astype(np.float32))

print('accuracy:', accuracy)
