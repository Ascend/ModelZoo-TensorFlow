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
import os
import tensorflow as tf

PLATFORM = 'gpu'
WORKSPACE_DIR = os.environ['PRJPATH']
SOURCE_DIR = os.path.join(WORKSPACE_DIR, 'src')
CHECKPOINT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoint')
LOG_DIR = os.path.join(WORKSPACE_DIR, 'log')

# Device
ALLOW_MULTI_DEVICE = True
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
NUM_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

# Session Config
def get_sess_cfg(sess_cfg=None):
    """get session config

    Args:
        sess_cfg ([sess_cfg], optional): session config. Defaults to None.
    Returns:
        sess_cfg
    """
    if sess_cfg is None:
        sess_cfg = tf.ConfigProto()
    sess_cfg.gpu_options.allow_growth = True
    sess_cfg.allow_soft_placement = True
