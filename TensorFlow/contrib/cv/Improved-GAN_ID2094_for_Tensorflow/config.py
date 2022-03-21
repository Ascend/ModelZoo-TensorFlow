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
from npu_bridge.npu_init import *
import argparse
import os

from model import Model
import datasets.hdf5_loader as dataset


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--dump_result', type=str2bool, default=False)
    # Model
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--norm_type', type=str, default='batch',
                        choices=['batch', 'instance', 'None'])
    parser.add_argument('--deconv_type', type=str, default='bilinear',
                        choices=['bilinear', 'nn', 'transpose'])
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=10000)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--output_save_step', type=int, default=1000)
    # learning
    parser.add_argument('--max_sample', type=int, default=5000, 
                        help='num of samples the model can see')
    parser.add_argument('--max_training_steps', type=int, default=45000)
    parser.add_argument('--learning_rate_g', type=float, default=1e-4)
    parser.add_argument('--learning_rate_d', type=float, default=1e-4)
    parser.add_argument('--update_rate', type=int, default=1)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument('--data_id', nargs='*', default=None)
    # }}}

    config = parser.parse_args()

    dataset_path = os.path.join('/cache/data', config.dataset.lower())
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path)

    img, label = dataset_train.get_data(dataset_train.ids[0])
    config.h = img.shape[0]
    config.w = img.shape[1]
    config.c = img.shape[2]
    config.num_class = label.shape[0] 

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_test

