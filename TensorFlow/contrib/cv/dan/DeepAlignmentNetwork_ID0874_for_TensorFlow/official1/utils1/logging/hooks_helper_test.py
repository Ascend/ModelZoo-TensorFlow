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

"""Tests for hooks_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import unittest

import tensorflow as tf

from official1.utils1.logging import hooks_helper


class BaseTest(unittest.TestCase):

  def test_raise_in_non_list_names(self):
    with self.assertRaises(ValueError):
      hooks_helper.get_train_hooks(
          'LoggingTensorHook, ProfilerHook', batch_size=256)

  def test_raise_in_invalid_names(self):
    invalid_names = ['StepCounterHook', 'StopAtStepHook']
    with self.assertRaises(ValueError):
      hooks_helper.get_train_hooks(invalid_names, batch_size=256)

  def validate_train_hook_name(self,
                               test_hook_name,
                               expected_hook_name,
                               **kwargs):
    returned_hook = hooks_helper.get_train_hooks([test_hook_name], **kwargs)
    self.assertEqual(len(returned_hook), 1)
    self.assertIsInstance(returned_hook[0], tf.train.SessionRunHook)
    self.assertEqual(returned_hook[0].__class__.__name__.lower(),
                     expected_hook_name)

  def test_get_train_hooks_logging_tensor_hook(self):
    test_hook_name = 'LoggingTensorHook'
    self.validate_train_hook_name(test_hook_name, 'loggingtensorhook')

  def test_get_train_hooks_profiler_hook(self):
    test_hook_name = 'ProfilerHook'
    self.validate_train_hook_name(test_hook_name, 'profilerhook')

  def test_get_train_hooks_examples_per_second_hook(self):
    test_hook_name = 'ExamplesPerSecondHook'
    self.validate_train_hook_name(test_hook_name, 'examplespersecondhook')

if __name__ == '__main__':
  (npu_sess, npu_shutdown) = init_resource()
  tf.test.main()
  shutdown_resource(npu_sess, npu_shutdown)
  close_session(npu_sess)

