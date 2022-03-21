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


""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from config import get_config, prepare_dirs, save_config
from data_loader import DataLoader
from trainer import HMRTrainer


def main(config):
    prepare_dirs(config)

    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        image_loader = data_loader.load()
        smpl_loader = data_loader.get_smpl_loader()

    trainer = HMRTrainer(config, image_loader, smpl_loader)
    save_config(config)
    trainer.train()


if __name__ == '__main__':
    config = get_config()
    main(config)