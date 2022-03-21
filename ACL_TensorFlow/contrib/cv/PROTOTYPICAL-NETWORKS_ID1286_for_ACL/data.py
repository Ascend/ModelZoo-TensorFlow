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
    Load Dataset
"""
from PIL import Image
import numpy as np
import os
import glob


class Data:
    """
    Dataset loading
    """
    def __init__(self, n_examples, im_height, im_width, datatype):
        #　root_dir = '/cache/data/omniglot'
        root_dir = './data/omniglot'
        split_path = os.path.join(root_dir, 'splits', datatype)
        with open(split_path, 'r') as split:
            classes = [line.rstrip() for line in split.readlines()]
        n_classes = len(classes)

        dataset = np.zeros([n_classes, n_examples, im_height, im_width], dtype=np.float32)

        for i, tc in enumerate(classes):
            alphabet, character, rotation = tc.split('/')
            rotation = float(rotation[3:])
            im_dir = os.path.join(root_dir, 'data', alphabet, character)
            im_files = sorted(glob.glob(os.path.join(im_dir, '*.png')))
            for j, im_file in enumerate(im_files):
                im = 1. - np.array(Image.open(im_file).rotate(rotation).resize((im_width, im_height)),
                                   np.float32, copy=False)
                dataset[i, j] = im

        self.n_classes = n_classes
        self.dataset = dataset
        self.shape = dataset.shape
