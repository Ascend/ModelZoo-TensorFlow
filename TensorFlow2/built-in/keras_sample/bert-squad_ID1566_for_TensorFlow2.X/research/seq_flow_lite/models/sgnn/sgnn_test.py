# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
#
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
#
# ==============================================================================

# Lint as: python3
"""Tests for seq_flow_lite.sgnn."""

import tensorflow as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from models import sgnn # import seq_flow_lite module


@test_util.run_all_in_graph_and_eager_modes
class SgnnTest(tf.test.TestCase):

  def test_preprocess(self):
    self.assertAllEqual(
        sgnn.preprocess(
            tf.constant([['Hello World!'], [u'浣犲ソ'],
                         [u'賲乇丨亘丕 亘丕賱毓丕賱賲']])),
        [['hello'.encode(), 'world!'.encode()], [u'浣犲ソ'.encode()],
         [u'賲乇丨亘丕'.encode(), u'亘丕賱毓丕賱賲'.encode()]])

  def test_get_ngram(self):
    tokens = tf.ragged.constant([['hello', 'world'], [u'浣犲ソ'],
                                 [u'賲乇丨亘丕', u'亘丕賱毓丕賱賲']])
    self.assertAllEqual(
        sgnn.get_ngrams(tokens, 3),
        [[
            b'^he', b'hel', b'ell', b'llo', b'lo$', b'^wo', b'wor', b'orl',
            b'rld', b'ld$'
        ], [u'^浣犲ソ'.encode(), u'浣犲ソ$'.encode()],
         [
             u'^賲乇'.encode(), u'賲乇丨'.encode(), u'乇丨亘'.encode(),
             u'丨亘丕'.encode(), u'亘丕$'.encode(), u'^亘丕'.encode(),
             u'亘丕賱'.encode(), u'丕賱毓'.encode(), u'賱毓丕'.encode(),
             u'毓丕賱'.encode(), u'丕賱賲'.encode(), u'賱賲$'.encode()
         ]])

  def test_project(self):
    ngrams = tf.ragged.constant([[b'^h', b'he', b'el', b'll', b'lo', b'o$'],
                                 [b'^h', b'hi', b'i$']])
    self.assertAllClose(
        sgnn.fused_project(ngrams, [5, 7], 0x7FFFFFFF),
        [[0.448691, -0.238499], [-0.037561, 0.080748]])
    self.assertAllClose(
        sgnn.fused_project(ngrams, [5, 7], 0x7FFFFFFF),
        sgnn.project(ngrams, [5, 7], 0x7FFFFFFF))

  def test_sgnn(self):
    self.assertAllClose(
        sgnn.sgnn(tf.constant([['hello'], ['hi']]), [3, 5, 7], 2),
        [[0.268503, 0.448691, -0.238499], [0.093143, -0.037561, 0.080748]])

  def test_keras_model(self):
    hparams = sgnn.Hparams(learning_rate=2e-4)
    model = sgnn.keras_model([1, 2, 3, 4], 2, [100, 50], hparams)
    self.assertIsNotNone(model)


if __name__ == '__main__':
  tf.test.main()
