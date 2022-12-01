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
# Copyright 2022 Huawei Technologies Co., Ltd
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


def generate_grid(lanes, cls_shape, image_shape, delete_lanes=None):    
    lanes = np.array(lanes)

    gt = np.zeros(cls_shape)

    for y in range(cls_shape[1]):
        yf = y / cls_shape[1]
        ly = int(lanes.shape[1] * yf)

        lx = lanes.T[ly,:]

        invalids = np.where(lx == -1)

        xf = lx / image_shape[1]
        x = np.round(xf * cls_shape[0]-1).astype(np.int)

        x[invalids] = cls_shape[0]-1

        if delete_lanes is not None:
            # delete unused lanes
            x = np.delete(x, (0,3))

        for i in range(len(x)):
            gt[x[i],y,i] = 1

    return gt
