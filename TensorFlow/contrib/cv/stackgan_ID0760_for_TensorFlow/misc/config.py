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
from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = 0
__C.Z_DIM = 100

# Demo/test options
__C.TEST = edict()
__C.TEST.LR_IMSIZE = 64
__C.TEST.HR_IMSIZE = 256
__C.TEST.NUM_COPY = 16
__C.TEST.BATCH_SIZE = 64
__C.TEST.NUM_COPY = 16
__C.TEST.PRETRAINED_MODEL = ''
__C.TEST.CAPTION_PATH = ''


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_COPY = 4
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600

__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.LR_DECAY_EPOCH = 50

__C.TRAIN.NUM_EMBEDDING = 4
__C.TRAIN.COND_AUGMENTATION = True
__C.TRAIN.B_WRONG = True

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# For Stage II training
__C.TRAIN.FINETUNE_LR = False
__C.TRAIN.FT_LR_RETIO = 0.1

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.NETWORK_TYPE = 'default'


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
