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
 
''' Config Proto '''
from npu_bridge.npu_init import *
import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'sphere64_casia_am_PFE'

# The folder to save log and model
log_base_dir = './log/'
# log_base_dir = "/home/ma-user/modelarts/outputs/train_url_0" + '/log/'

# The interval between writing summary
summary_interval = 100

# Training dataset path
# train_dataset_path = "data/list_casia_mtcnncaffe_aligned_nooverlap.txt"
train_dataset_path = "data/list_casia_mtcnncaffe_aligned_nooverlap_npu.txt"
# train_dataset_path = "data/list_casia_mtcnncaffe_aligned_nooverlap_yzy.txt"
# train_dataset_path = "data/list_casia_mtcnncaffe_aligned_nooverlap_yzy_new.txt"


# Target image size for the input of network
image_size = [112, 96]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    ['center_crop', (112, 96)],
    ['random_flip'],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['center_crop', (112, 96)],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# The network architecture
embedding_network = "models/sphere_net_PFE.py"

# The network architecture
uncertainty_module = "models/uncertainty_module.py"

# Number of dimensions in the embedding space
embedding_size = 512


####### TRAINING STRATEGY #######

# Base Random Seed
base_random_seed = 0

# Number of samples per batch
batch_format = {
    'size': 256,
    'num_classes': 64,
}


# Number of batches per epoch
epoch_size = 1000

# Number of epochs
num_epochs = 3

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 1e-3
learning_rate_schedule = {
    0:      1 * lr,
    2000:   0.1 * lr,
}

# Restore model
# restore_model = './pretrained/sphere64_casia_am'
restore_model = 'pretrained/sphere64_casia_am'

# Keywords to filter restore variables, set None for all
restore_scopes = ['SphereNet/conv', 'SphereNet/Bot']

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0


