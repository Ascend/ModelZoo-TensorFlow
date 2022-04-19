# Copyright 2022 Huawei Technologies Co., Ltd
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

import copy
import os

import tensorflow as tf
from src.utils.world import world


class _AdversarialTrainHelper:
    """A helper for adversarial training.

    In each step, the generator and discriminator will produce losses.
    The helper determines whether to update the G and D in the next
    iteration. For example, in step `i`, if the D loss (already evaluated) 
    is found very small, which means that D might be too strong and should 
    halt for some steps, the helper will filter out the discriminator ops in
    step `i+1` to keep the discriminator, then evaluate again in step `i+1` 
    to determine whether to update G and D in step `i+2`.

    Example:
        >>> # Define the tf ops
        >>> helper = _AdversarialTrainHelper()
        >>> g_train_op, d_train_op = define_train()
        >>> d_train_op_dict = {'d_op': d_train_op}
        >>> g_train_op_dict = {'g_op': g_train_op}
        >>> d_loss = compute_d_loss(fake, real)
        >>> ...
        >>> # in step i=0
        >>> i = 0
        >>> g_train_op_dict_real, d_train_op_dict_real = helper.filter(g_train_op_dict, d_train_op_dict)
        >>> _, _, d_loss_eval = sess.run([g_train_op_dict_real, d_train_op_dict_real, d_loss])
        >>> helper.update_status(d_loss_eval, i+1)
        >>> ...
        >>> # in step i=1
        >>> i = 1
        >>> # decide whether to update G and D in step 1 according to the result in step 0.
        >>> g_train_op_dict_real, d_train_op_dict_real = helper.filter(g_train_op_dict, d_train_op_dict)
        >>> _, _, d_loss_eval = sess.run([g_train_op_dict, d_train_op_dict, d_loss])
        >>> # update status with the d_loss and step index.
        >>> helper.update_status(d_loss_eval, i+1)
    """
    def __init__(self):
        self._called_once = False
        self._info = dict()

    @property
    def info(self):
        return self._info

    def filter(self, g_ops_in, d_ops_in, *args, **kwargs):
        if not self._called_once:
            # For the first time, we must run all the operations on NPU
            # to construct the whole graph. It is regardless of  the update 
            # strategy.
            g_update, d_update = True, True
            self._called_once = True
        else:
            # Once called and initialized, use the configured strategy to
            # check whether to update G and D.
            g_update, d_update = self.check_state()

        # Post validation to make sure that at least one of G and D should
        # update.
        g_update, d_update = self.post_validation(g_update, d_update)

        # Save the decision
        self._info = dict(
            g_update=g_update,
            d_update=d_update
        )

        g_ops = dict(**g_ops_in)
        if not g_update:
            g_ops.pop('g_train')

        d_ops = dict(**d_ops_in)
        if not d_update and 'd_train' in d_ops_in:
            d_ops.pop('d_train')

        return g_ops, d_ops

    def post_validation(self, g_update, d_update):
        # Abnormal states when both g_update and d_update are false
        if (not g_update) and (not d_update):
            g_update = True
            d_update = False
        return g_update, d_update

    def check_state(self):
        # This is where specific strategy should implement how to make decisions
        # on the whether to update G and D.
        raise NotImplementedError

    def update_status(self, *args, **kwargs):
        # Record the step and the criteria value.
        raise NotImplementedError

    def not_initialized(self):
        raise ValueError(f'Helper has not been initialized.')


class ByPassTrainHelper(_AdversarialTrainHelper):
    """A bypass train helper.
    
    The ops will not be filtered at all.
    """
    def __init__(self, use_adv=False):
        super().__init__()
        self.use_adv = use_adv

    def check_state(self):
        # g_update always True
        # d_update according to self.use_adv
        return True, self.use_adv

    def update_status(self, *args, **kwargs):
        pass


class AdaptiveTrainHelper(_AdversarialTrainHelper):
    """An adaptive train helper given the loss values.
    """
    def __init__(self, d_threshold, g_threshold=None):
        super().__init__()
        self.d_threshold = d_threshold
        self.g_threshold = g_threshold
        self.previous_d_loss = None
        self.previous_step = None
        self.d_warmstarted = False

    def update_status(self, loss=None, step=None):
        self.previous_d_loss = loss
        self.previous_step = step

    def check_state(self):
        if self.previous_d_loss is None:
            self.not_initialized()

        d_update = self.previous_d_loss > self.d_threshold
        if not self.d_warmstarted:
            # Don't ever update generator when discriminator is not yet that strong.
            # Once the discriminator is at first time strong enough, apply the dynamic update.
            g_update = False
            if not d_update:
                self.d_warmstarted = True
        elif self.g_threshold is None:
            g_update = True
        else:
            g_update = self.previous_d_loss < self.g_threshold

        return g_update, d_update


class FixedStepTrainHelper(_AdversarialTrainHelper):
    """A train helper with fixed interval.
    """
    def __init__(self, g_update_interval=-1, d_update_interval=-1):
        super().__init__()
        self.g_update_interval = g_update_interval
        self.d_update_interval = d_update_interval
        self.previous_step = None

    def update_status(self, loss=None, step=None):
        self.previous_step = step

    def check_state(self):
        if self.previous_step is None:
            self.not_initialized()

        g_update = (self.previous_step + 1) % self.g_update_interval == 0
        d_update = (self.previous_step + 1) % self.d_update_interval == 0

        return g_update, d_update


def build_adversarial_train_helper(cfg):
    """Build corresponding train helper given the configuration.
    
    Args:
        cfg: yacs node, global configuration.

    Returns:
        helper instance.
    """
    if cfg.loss.adversarial.loss_weight > 0.:
        if cfg.loss.adversarial.adaptive_strategy:
            helper = AdaptiveTrainHelper(cfg.loss.adversarial.d_balance)
        elif cfg.loss.adversarial.g_update_interval > 1 or cfg.loss.adversarial.d_update_interval > 1:
            if cfg.loss.adversarial.g_update_interval > 1 and cfg.loss.adversarial.d_update_interval > 1:
                raise ValueError('Either g update interval or d update interval should be 1.')
            helper = FixedStepTrainHelper(cfg.loss.adversarial.g_update_interval,
                                          cfg.loss.adversarial.d_update_interval)
        else:
            # no helper
            helper = ByPassTrainHelper(use_adv=True)
    else:
        # no helper
        helper = ByPassTrainHelper(use_adv=False)

    return helper
