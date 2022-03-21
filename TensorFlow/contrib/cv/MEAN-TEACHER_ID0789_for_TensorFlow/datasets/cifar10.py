# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
import os

import numpy as np

from .utils import random_balanced_partitions, random_partitions


class Cifar10ZCA:
    VALIDATION_SET_SIZE = 5000  # 10% of the training set
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', data_path='data', test_phase=False):
        #self.data_path = os.path.join(data_path, 'images', 'cifar', 'cifar10', 'cifar10_gcn_zca_v2.npz')
        self.data_path = os.path.join(data_path, 'cifar10_gcn_zca_v2.npz')

        random = np.random.RandomState(seed=data_seed)
        self._load()

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)

    def _load(self):
        print(self.data_path)
        print(os.listdir())
        print('codedir:',os.path.dirname(__file__))
        print('work_dir', os.getcwd())
        file_data = np.load(self.data_path)
        self._train_data = self._data_array(50000, file_data['train_x'], file_data['train_y'])
        self._test_data = self._data_array(10000, file_data['test_x'], file_data['test_y'])

    def _data_array(self, expected_n, x_data, y_data):
        array = np.zeros(expected_n, dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        array['x'] = x_data
        array['y'] = y_data
        return array

    def _validation_and_training(self, random):
        return random_partitions(self._train_data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self._test_data, self._train_data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])

