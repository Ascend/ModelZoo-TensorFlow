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
import glob
import os
from PIL import Image


def load_images(path, image_size):
    train_all = sorted(glob.glob(os.path.join(path, "train/*.jpg")))
    test_all = sorted(glob.glob(os.path.join(path, "val/*.jpg")))

    train_input = []
    test_input = []
    train_output = []
    test_output = []

    for img in train_all:
        full_image = Image.open(img)
        full_image = np.asarray(full_image.resize((2 * image_size, image_size), Image.BICUBIC))

        # in maps dataset,the input and output merge to one image
        # and the output is the left part
        train_output.append(full_image[:, :full_image.shape[1] // 2, :] / 255.)
        train_input.append(full_image[:, full_image.shape[1] // 2:, :] / 255.)

    for img in test_all:
        full_image = Image.open(img)
        full_image = np.asarray(full_image.resize((2 * image_size, image_size), Image.BICUBIC))

        test_output.append(full_image[:, :full_image.shape[1] // 2, :] / 255.)
        test_input.append(full_image[:, full_image.shape[1] // 2:, :] / 255.)

    # need to normalize to [-1,1]
    return np.asarray(train_input) * 2 - 1, np.asarray(train_output) * 2 - 1, \
           np.asarray(test_input) * 2 - 1, np.asarray(test_output) * 2 - 1
