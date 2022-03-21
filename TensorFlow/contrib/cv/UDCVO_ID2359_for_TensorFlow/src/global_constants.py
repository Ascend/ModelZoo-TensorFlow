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

'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf
@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
'''
  Global constants
'''
# Input image dimensions
N_BATCH             = 8
N_HEIGHT            = 320
N_WIDTH             = 768
N_CHANNEL           = 3
# Network Hyperparameters
N_EPOCH             = 50
LEARNING_RATES      = [1.2e-4, 0.6e-4, 0.3e-4]
LEARNING_BOUNDS     = [18, 24]
LEARNING_RATES_TXT  = '1.2e-4,0.6e-4,0.3e-4'
LEARNING_BOUNDS_TXT = '18,24'
MIN_Z               = 1.50
MAX_Z               = 100.00
POSE_NORM           = 'frobenius'
ROT_PARAM           = 'exponential'
# Preprocessing
OCC_THRESHOLD       = 1.5
OCC_KSIZE           = 7
# Network Structure
NET_TYPE            = 'vggnet11'
IM_FILTER_PCT       = 0.75
SZ_FILTER_PCT       = 0.25
# Weights for loss function
W_PH                = 1.00
W_CO                = 0.20
W_ST                = 0.80
W_SM                = 0.01
W_SZ                = 0.20
W_PC                = 0.10
# Checkpoint paths
CHECKPOINT_PATH     = 'log'
RESTORE_PATH        = ''
N_CHECKPOINT        = 5000
N_SUMMARY           = 1000
OUTPUT_PATH         = 'out'
# Hardware settings
N_THREAD            = 8
EPSILON             = 1e-10
