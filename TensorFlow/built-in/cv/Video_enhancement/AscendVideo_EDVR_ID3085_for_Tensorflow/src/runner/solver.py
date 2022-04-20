# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Copyright (c) 2022 Huawei Technologies Co., Ltd
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


from .lr_schedule import build_schedule
from .optimizer import build_optimizer


class Solver(object):
    """Solver class including optimizer and learning-rate schedule.

    Args:
        lr_cfg: yacs node, learning-rate schedule.
        opt_cfg: yacs node, optimizer config.
        device: str, valid options: ['npu', 'cpu']
        is_distributed: boolean, whether used in distributed training.
        mix_precision: boolean, whether used mix precision during training.
        loss_scale: boolean, whether use loss scaling to compensate the
            precision loss during dtype conversion.
    """
    def __init__(self, lr_cfg, opt_cfg, device, is_distributed, mix_precision, 
                 loss_scale):
        self.lr_schedule = build_schedule(lr_cfg)
        self.opt = build_optimizer(self.lr_schedule.lr, opt_cfg, 
                                   device, 
                                   is_distributed, 
                                   mix_precision, 
                                   loss_scale)
        self.total_step = sum(lr_cfg.total_steps)

    def update_lr(self):
        """Update learning rate based on schedule and step.
        """
        return self.lr_schedule()

    @property
    def lr(self):
        """Returns learning rate placeholder.
        """
        return self.lr_schedule.lr

    @property
    def cur_lr(self):
        """Returns current learning rate.
        """
        return self.lr_schedule.cur_lr


def build_solver(lr_cfg, optimizer_cfg, mix_precision, loss_scale, device, 
                 is_distributed):
    """Build solver for training.

    Args:
        lr_cfg: yacs node, learning-rate schedule.
        optimizer_cfg: yacs node, optimizer config.
        device: str, valid options: ['npu', 'cpu']
        is_distributed: boolean, whether used in distributed training.
        mix_precision: boolean, whether used mix precision during training.
        loss_scale: boolean, whether use loss scaling to compensate the
            precision loss during dtype conversion.
    
    Return:
        A solver instance.
    """
    assert device in ['npu', 'cpu']

    return Solver(lr_cfg, 
                  optimizer_cfg, 
                  device, 
                  is_distributed, 
                  mix_precision, 
                  loss_scale)
