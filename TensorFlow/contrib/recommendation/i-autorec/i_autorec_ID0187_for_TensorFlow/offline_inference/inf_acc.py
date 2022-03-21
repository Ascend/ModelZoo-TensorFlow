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

input_R = np.fromfile('bin_data/input_R/a.bin', dtype=np.float32)
input_mask_R = np.fromfile('bin_data/input_mask_R/a.bin', dtype=np.float32)
user_list = np.fromfile('bin_data/user/a.bin', dtype=np.float32)
item_list = np.fromfile('bin_data/item/a.bin', dtype=np.float32)
num_ratings = np.fromfile('bin_data/ratings/a.bin', dtype=np.int32)

inf = np.fromfile('PATH/autorec_base_output_0.bin', dtype=np.float32)
inf = inf.reshape((-1, 80))
input_R = input_R.reshape((-1, 80))
input_mask_R = input_mask_R.reshape((-1, 80))

Estimated_R = inf.clip(min=1, max=5)
for user in user_list:
  for item in item_list:
    if input_mask_R[user, item] == 1:
      Estimated_R[user, item] = 3

pre_numerator = np.multiply((Estimated_R - input_R), input_mask_R)
numerator = np.sum(np.square(pre_numerator))
denominator = num_ratings
RMSE = np.sqrt(numerator / float(denominator[0]))

print('RMSE:', RMSE)
