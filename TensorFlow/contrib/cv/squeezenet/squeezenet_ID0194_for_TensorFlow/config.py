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

import os
os.system("pip install easydict")
import tensorflow as tf
import pathlib
from easydict import EasyDict

_C = EasyDict()

# dataconfig
_C.DATA_AUGMENTATION = True
_C.NUM_DATA_WORKERS = 8
#   Model Arts Config 
_C.CACHE_DATA_FILENAMES_CACHE_PATH = 'data/imagenet/'
pathlib.Path(_C.CACHE_DATA_FILENAMES_CACHE_PATH).mkdir(parents=True, exist_ok=True)

_C.TRAINING_CKPT_FILE_DIR_NAMES = 'data/'
pathlib.Path(_C.TRAINING_CKPT_FILE_DIR_NAMES).mkdir(parents=True, exist_ok=True)

# Moddel Config
_C.bnmomemtum=0.9
_C.LR=1e-3

_C.IMAGE_SIZE = 224
# cfg.IMAGE_SIZE

# Train Config
# tf record --per class 500 train--  --per class 50 val, -- per class 50 test class = 200
_C.BATCH_SIZE = 64
_C.EPOCH = 2 #80 * 60

_C.STEPS_PER_EPOCH = 401 // 4
_C.EPOCHS = 3 #100000000
_C.VALIDATION_STEPS = 10