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
from npu_bridge.npu_init import *
import os


def check(dirname):
    """This function creates a directory
    in case it doesn't exist"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


# The project directory
EVAL_DIR = os.path.dirname(os.path.realpath(__file__))

# Folder with datasets
DATASETS_ROOT = check(os.path.join(EVAL_DIR, 'Datasets'))

# Where the checkpoints are stored
CKPT_ROOT = check(os.path.join(EVAL_DIR, 'archive\\'))

# Where the logs are stored
LOGS = check(os.path.join(EVAL_DIR, 'Logs\\Experiments\\'))

# Where the algorithms visualizations are dumped
RESULTS_DIR = check(os.path.join(EVAL_DIR, 'Results\\'))

# Where the imagenet weights are located
INIT_WEIGHTS_DIR = check(os.path.join(EVAL_DIR, '../Weights_imagenet\\'))

# Where the demo images are located
DEMO_DIR = check(os.path.join(EVAL_DIR, 'Demo\\'))

