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


"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
from imageio import imsave

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((int(h*nh), int(w*nw), 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = x

    imsave(save_path, img)