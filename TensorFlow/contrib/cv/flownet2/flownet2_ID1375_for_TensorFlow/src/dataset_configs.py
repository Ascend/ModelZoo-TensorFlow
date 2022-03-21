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
"""
Add dataset configurations here. Each dataset must have the following structure:

NAME = {
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    ITEMS_TO_DESCRIPTIONS: {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    SIZES: {
        'train': int,
        'validate': int,    (optional)
        ...
    },
    BATCH_SIZE: int,
    PATHS: {
        'train': '',
        'validate': '', (optional)
        ...
    }
}
"""

"""
note that one step = one batch of data processed, ~not~ an entire epoch
'coeff_schedule_param': {
    'half_life': 50000,         after this many steps, the value will be i + (f - i)/2
    'initial_coeff': 0.5,       initial value
    'final_coeff': 1,           final value
},
"""

FLYING_CHAIRS_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 384,
    'IMAGE_WIDTH': 512,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    'SIZES': {
        'train': 22232,
        'validate': 640,
        'sample': 8,
    },
    'BATCH_SIZE': 2,
    'PATHS': {
        'train': './data/tfrecords/fc_train.tfrecords',
        'validate': './data/tfrecords/fc_val.tfrecords',
        'sample': './data/tfrecords/fc_sample.tfrecords',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 320,
        'crop_width': 448,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            'noise': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0.03,
                'spread': 0.03,
                'prob': 1.0,
            },
        },
    #     # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    },
}
