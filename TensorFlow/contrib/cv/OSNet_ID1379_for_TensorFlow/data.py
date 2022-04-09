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
import math
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img
from sklearn.utils import shuffle as shuffle_tuple
from utils.general import seq


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, num_classes, shuffle=False, augment=False):
        self.x, self.y = x_set, y_set
        self.total_num_image = len(x_set)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.shuffle:
            batch_x, batch_y = shuffle_tuple(batch_x, batch_y)

        if self.augment:
            batch_x = np.array([np.asarray(load_img(file_path)).astype(np.uint8) for file_path in batch_x]).astype(np.uint8)
            batch_x = seq.augment_images(batch_x)
            batch_x = batch_x / 255.
        else:
            batch_x = np.array([np.asarray(load_img(file_name)) / 255. for file_name in batch_x])

        batch_x = (batch_x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch_y = to_categorical(np.array(batch_y), num_classes=self.num_classes)

        return batch_x, batch_y