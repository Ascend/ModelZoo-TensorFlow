# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Config template to train Retinanet."""
from npu_bridge.npu_init import *

from configs import detection_config
from hyperparameters import params_dict

# pylint: disable=line-too-long
RETINANET_CFG = params_dict.ParamsDict(detection_config.DETECTION_CFG)
RETINANET_CFG.override({
    'type': 'retinanet',
    'architecture': {
        'parser': 'retinanet_parser',
        'backbone': 'resnet',
        'multilevel_features': 'fpn',
    },
    'retinanet_parser': {
        'output_size': [640, 640],
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'aug_policy': '',
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'regenerate_source_id': False,
    },
    'retinanet_head': {
        'anchors_per_location': None,  # Param no longer used.
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'retinanet_loss': {
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
        'normalizer_momentum': 0.0,
    },
}, is_strict=False)

RETINANET_RESTRICTIONS = [
]
# pylint: enable=line-too-long

