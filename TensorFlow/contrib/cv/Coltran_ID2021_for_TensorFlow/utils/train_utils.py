"""
train
"""
# coding=utf-8
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

"""Utils for training."""

import os
from absl import logging
import numpy as np
import tensorflow as tf


def dataset_with_strategy(dataset_fn, strategy):
    """
        Args:

        Returns:
        """
    if strategy:
        return strategy.experimental_distribute_datasets_from_function(dataset_fn)
    else:
        return dataset_fn(None)


def with_strategy(fn, strategy):
    """
        Args:

        Returns:
        """
    logging.info(strategy)
    if strategy:
        with strategy.scope():
            return fn()
    else:
        return fn()


def save_nparray_to_disk(filename, nparray):
    """
        Args:

        Returns:
        """
    fdir, _ = os.path.split(filename)
    if not tf.io.gfile.exists(fdir):
        tf.io.gfile.makedirs(fdir)
    with tf.io.gfile.GFile(filename, 'w') as f:
        np.save(f, nparray)

