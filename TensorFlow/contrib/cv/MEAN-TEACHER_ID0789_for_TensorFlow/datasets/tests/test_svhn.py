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
import numpy as np

from ..svhn import SVHN


def test_supervised_testing():
    svhn = SVHN(test_phase=True)
    assert_shapes(svhn.training, 73257)
    assert_label_range(svhn.training, range(10))
    assert_shapes(svhn.evaluation, 26032)
    assert_label_range(svhn.evaluation, range(10))


def test_semisupervised_testing():
    svhn = SVHN(test_phase=True, n_labeled=100)
    assert_shapes(svhn.training, 73257)
    assert_label_range(svhn.training, range(-1, 10))
    assert_label_distribution(svhn.training, [73157] + [10] * 10)
    assert_shapes(svhn.evaluation, 26032)
    assert_label_range(svhn.evaluation, range(10))


def test_extra_unlabeled_testing():
    svhn = SVHN(test_phase=True, n_labeled=100, n_extra_unlabeled=10000)
    assert_shapes(svhn.training, 83257)
    assert_label_range(svhn.training, range(-1, 10))
    assert_label_distribution(svhn.training, [83157] + [10] * 10)
    assert_shapes(svhn.evaluation, 26032)
    assert_label_range(svhn.evaluation, range(10))


def test_supervised_validation():
    svhn = SVHN()
    assert_shapes(svhn.training, 65932)
    assert_label_range(svhn.training, range(10))
    assert_shapes(svhn.evaluation, 7325)
    assert_label_range(svhn.evaluation, range(10))


def test_semisupervised_validation():
    svhn = SVHN(n_labeled=100)
    assert_shapes(svhn.training, 65932)
    assert_label_range(svhn.training, range(-1, 10))
    assert_label_distribution(svhn.training, [65832] + [10] * 10)
    assert_shapes(svhn.evaluation, 7325)
    assert_label_range(svhn.evaluation, range(10))


def test_extra_unlabeled_validation():
    svhn = SVHN(n_labeled=100, n_extra_unlabeled=10000)
    assert_shapes(svhn.training, 75932)
    assert_label_range(svhn.training, range(-1, 10))
    assert_label_distribution(svhn.training, [75832] + [10] * 10)
    assert_shapes(svhn.evaluation, 7325)
    assert_label_range(svhn.evaluation, range(10))


def assert_shapes(data, n_expected_examples):
    assert data['x'].shape == (n_expected_examples, 32, 32, 3)
    assert data['y'].shape == (n_expected_examples,)


def assert_label_range(data, expected_range):
    assert np.min(data['y']) == min(expected_range)
    assert np.max(data['y']) == max(expected_range)


def assert_label_distribution(data, expected_distribution):
    label_distribution = np.bincount(data['y'] + 1)
    assert label_distribution.tolist() == expected_distribution

