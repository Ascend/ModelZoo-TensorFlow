#! /usr/bin/env python
# coding=utf-8
#
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
#
# @Author: Longxing Tan, tanlongxing888@163.com

import npu_device
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Optimizer(object):
    def __init__(self, optimizer_method='adam'):
        self.optimizer_method = optimizer_method

    def __call__(self):
        if self.optimizer_method == 'adam':
            return tf.keras.optimizers.Adam()
        elif self.optimizer_method == 'rmsprop':
            return tf.keras.optimizers.RMSprop()
        elif self.optimizer_method == 'sgd':
            return tf.keras.optimizers.SGD()
        else:
            raise ValueError('Unsupported optimizer {}'.format(self.optimizer_method))


class LrScheduler(object):
    def __init__(self, total_steps, params, scheduler_method='cosine'):
        if scheduler_method == 'step':
            self.scheduler = Step(total_steps, params)
        elif scheduler_method == 'cosine':
            self.scheduler = Cosine(total_steps, params)
        self.step_count = 0
        self.total_steps = total_steps

    def step(self):
        self.step_count += 1
        lr = self.scheduler(self.step_count)
        return lr

    def plot(self):
        lr = []
        for i in range(self.total_steps):
            lr.append(self.step())
        plt.plot(range(self.total_steps), lr)
        plt.show()


class Step(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, params):
        # create the step learning rate with linear warmup
        super(Step, self).__init__()
        self.total_steps = total_steps
        self.params = params

    def __call__(self, global_step):
        warmup_lr = self.params['warmup_learning_rate']
        warmup_steps = self.params['warmup_steps']
        init_lr = self.params['init_learning_rate']
        lr_levels = self.params['learning_rate_levels']
        lr_steps = self.params['learning_rate_steps']
        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

        for next_learning_rate, start_step in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)

        return learning_rate


class Cosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, params):
        # create the cosine learning rate with linear warmup
        super(Cosine, self).__init__()
        self.total_steps = total_steps
        self.params = params

    def __call__(self, global_step):
        init_lr = self.params['init_learning_rate']
        warmup_lr = self.params['warmup_learning_rate'] if 'warmup_learning_rate' in self.params else 0.0
        warmup_steps = self.params['warmup_steps']
        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        cosine_learning_rate = init_lr * (
                    tf.cos(np.pi * (global_step - warmup_steps) / (self.total_steps - warmup_steps)) + 1.0) / 2.0
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, cosine_learning_rate)
        return learning_rate
