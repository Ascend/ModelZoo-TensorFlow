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

def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    moon_data = np.load('moon_data.npz')
    x_s = moon_data['x_s']
    y_s = moon_data['y_s']
    x_t = moon_data['x_t']
    return x_s, y_s, x_t

def generate_grid_point():
    x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
    y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    return xx, yy

x_s, y_s, x_t = load_data()
xx, yy = generate_grid_point()

data =  np.c_[xx.ravel(), yy.ravel()]
# [1000, 2]
# [2]
for i, x in enumerate(data):
    print(i)
    x.astype(np.float32).tofile(f"point/{i}.bin")