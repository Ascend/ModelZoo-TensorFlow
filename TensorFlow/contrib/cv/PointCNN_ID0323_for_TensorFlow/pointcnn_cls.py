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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
import tensorflow as tf
from pointcnn import PointCNN
from npu_bridge.npu_init import *


class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
        self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)
        self.logits = pf.dense(self.fc_layers[-1], setting.num_class, 'logits',
                               is_training, with_bn=False, activation=None)
