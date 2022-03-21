# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf
import math
import numpy as np

def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr

def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    lr_ini = lr
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []

    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        elif i < 330000:
            lr = lr_ini
        elif i < 520000 and i >= 330000:
            base_lr = lr_ini / 10
            lr = base_lr
        elif i < 625000 and i >= 520000:
            base_lr = lr_ini / 100
            lr = base_lr
        elif i > 625000:
            base_lr = lr_ini / 1000
            lr = base_lr
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)

class HyperParams:
    def __init__(self, args):
        self.args=args
        nsteps_per_epoch = self.args.num_training_samples // self.args.global_batch_size   #1281025/256=5004
        self.args.nsteps_per_epoch = nsteps_per_epoch
        if self.args.max_epochs:
            nstep = nsteps_per_epoch * self.args.max_epochs
        else:
            nstep = self.args.max_train_steps
        self.args.nstep = nstep

        #+++
        self.args.save_sammary_steps = nsteps_per_epoch
        self.args.save_checkpoints_steps = nsteps_per_epoch

        self.cos_lr = warmup_cosine_annealing_lr(self.args.lr, nsteps_per_epoch, 0, self.args.T_max, self.args.T_max, 0.0)

    #+++
    def get_hyper_params(self):
        hyper_params = {}
        hyper_params['learning_rate'] = self.get_learning_rate()

        return hyper_params

    def get_learning_rate(self): 
        global_step = tf.train.get_global_step()
    
        learning_rate = tf.gather(tf.convert_to_tensor(self.cos_lr), global_step)

        learning_rate = tf.identity(learning_rate, 'learning_rate')

        return learning_rate

