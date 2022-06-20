#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
#
#
# Copyright © 2017 bily          Huazhong University of Science and Technology
# Copyright © 2018 Anfeng He     Microsoft Research Asia. University of Science and Technology of China.
# Copyright (c) Microsoft. All rights reserved.
#
# Distributed under terms of the MIT license.

"""Train the color model in the SiamFC paper from scratch"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from configuration import LOG_DIR
from train_siamese_model import ex

if __name__ == '__main__':

  RUN_NAME = 'SA-Siam-Appearance'
  ex.run(config_updates={'train_config': {'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME), },
                         'track_config': {'log_dir': osp.join(LOG_DIR, 'track_model_inference', RUN_NAME), },
                         'model_config': {'sa_siam_config': {'en_appearance': True, }, },
                         },
         options={'--name': RUN_NAME,
                  '--force': True,
                  '--enforce_clean': False,
                  })

