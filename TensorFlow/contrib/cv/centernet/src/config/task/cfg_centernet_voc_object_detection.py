#
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
NAME = 'centernet_voc'
MAX_OBJECTS = 150
DATA_MEAN = [0.471, 0.448, 0.408]  # From MSCOCO
DATA_STD = [0.234, 0.239, 0.242]  # From MSCOCO
RESIZE_IMG_SHAPE = [512, 512]
# Data Augmentation
AUGMENTOR_CONFIG = {
    'data_format': 'channels_last',
    'output_shape': RESIZE_IMG_SHAPE,
    'constant_values': 0.,
    'pad_truth_to': MAX_OBJECTS
}
VALIDATION_AUGMENTOR_CONFIG = {
    'data_format': 'channels_last',
    'output_shape': RESIZE_IMG_SHAPE,
    'constant_values': 0.,
    'pad_truth_to': MAX_OBJECTS
}

# Train
IS_TRAINING = True
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3