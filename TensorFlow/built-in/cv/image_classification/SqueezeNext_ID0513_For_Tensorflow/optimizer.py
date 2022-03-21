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
from npu_bridge.npu_init import *
import tensorflow as tf
import tools

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt
slim = tf.contrib.slim

class PolyOptimizer(object):

    def __init__(self, training_params):
        self.base_lr = training_params['base_lr']
        self.warmup_steps = training_params['warmup_iter']
        self.warmup_learning_rate = training_params['warmup_start_lr']
        self.power = 2.0
        self.momentum = 0.9

    def optimize(self, loss, training, total_steps):
        '\n        Momentum optimizer using a polynomial decay and a warmup phas to match this\n        prototxt: https://github.com/amirgholami/SqueezeNext/blob/master/1.0-SqNxt-23/solver.prototxt\n        :param loss:\n            Loss value scalar\n        :param training:\n            Whether or not the model is training used to prevent updating moving mean of batch norm during eval\n        :param total_steps:\n            Total steps of the model used in the polynomial decay\n        :return:\n            Train op created with slim.learning.create_train_op\n        '
        with tf.name_scope('PolyOptimizer'):
            global_step = tools.get_or_create_global_step()
            learning_rate_schedule = tf.train.polynomial_decay(learning_rate=self.base_lr, global_step=global_step, decay_steps=total_steps, power=self.power)
            learning_rate_schedule = tools.warmup_phase(learning_rate_schedule, self.base_lr, self.warmup_steps, self.warmup_learning_rate)
            tf.summary.scalar('learning_rate', learning_rate_schedule)
            optimizer = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate_schedule, self.momentum))
            return slim.learning.create_train_op(loss, optimizer, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N, update_ops=(None if training else []))
