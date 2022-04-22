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
import logging, os

logging.basicConfig(format='[%(asctime)s] %(name)s: %(message)s', level=logging.INFO)


RANDOM_SEED = None

SUMMARY_INTERVAL        = 'auto' # int(of iter) or 'auto'
IMAGE_SUMMARY_INTERVAL  = 'auto' # int(of iter) or 'auto'


ROOT_DIR = "."   # change this to the project folder


WEIGHT_ROOT_DIR       = ROOT_DIR+"/SymNet_NPU/weights/"
LOG_ROOT_DIR          = ROOT_DIR+"/SymNet_NPU/output_log/"
DATA_ROOT_DIR         = ROOT_DIR+"/data"


CZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-original',
    'UT': 'ut-zap50k-original',
}

GCZSL_DS_ROOT = {
    'MIT': DATA_ROOT_DIR+'/mit-states-natural',
    'UT':  DATA_ROOT_DIR+'/ut-zap50k-natural',
}

GRADIENT_CLIPPING = 5


# if not os.path.exists(WEIGHT_ROOT_DIR):
#     os.makedirs(WEIGHT_ROOT_DIR)
# if not os.path.exists(LOG_ROOT_DIR):
#     os.makedirs(LOG_ROOT_DIR)