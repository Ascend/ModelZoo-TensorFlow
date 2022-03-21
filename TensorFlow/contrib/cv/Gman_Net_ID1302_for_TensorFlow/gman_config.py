"""
config
"""
# coding=utf-8
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

import tensorflow as tf

import gman_flags as df

# Configure for training
CONFIG_TRAINING_TRAIN_RESTORE = "train_restore"


def config_load_config():
    """

    Returns:

    """
    config = {CONFIG_TRAINING_TRAIN_RESTORE: df.FLAGS.train_restore}
    return config


def config_update_config(config):
    """

    Args:
        config:

    Returns:

    """
    if not config[CONFIG_TRAINING_TRAIN_RESTORE]:
        config[CONFIG_TRAINING_TRAIN_RESTORE] = True


if __name__ == '__main__':
    pass
