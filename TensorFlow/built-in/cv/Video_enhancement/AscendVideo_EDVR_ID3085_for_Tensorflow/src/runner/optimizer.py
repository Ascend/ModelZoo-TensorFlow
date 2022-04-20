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
import tensorflow as tf


def build_optimizer(lr, opt_cfg, device, is_distributed, mix_precision, loss_scale):
    """Build optimizer

    Args:
        lr: learning rate schedule instance.
        opt_cfg: dict, specifying the optimizer configuration.
        device: str, specifying the device type. Possible choices in ('npu', 'cpu').
        is_distributed: boolean, whether in distributed learning.
        mix_precision: boolean, whether to use mix precisioin.
        loss_scale: str, specifying the strategy to apply loss scaling. 
            Possible choices could be `off`: do not use loss scaling. 
            `d`: dynamic loss scaling, and `f*`: fixed loss scaling,
            where `*` can be converted to an integer that specifies the 
            scale factor, `2^(int(*))`.
    
    Returns:
        An optimizer instance.
    """
    opt_type = opt_cfg.type.lower()

    if opt_type == 'adam':
        beta1 = opt_cfg.get('beta1', 0.9)
        beta2 = opt_cfg.get('beta2', 0.999)
        epsilon = opt_cfg.get('epsilon', 1e-08)
        opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    elif opt_type == 'momentum':
        momentum = opt_cfg.get('momentum', 0.9)
        opt = tf.train.MomentumOptimizer(lr, momentum=momentum)
    else:
        raise KeyError('Unkown type {}'.format(opt_type))

    if device == 'npu':
        return npu_optimizer_wrapper(opt, mix_precision, loss_scale, is_distributed)
    else:
        return opt


def npu_optimizer_wrapper(opt, mix_precision, loss_scale, is_distributed=False):
    """A wrapper function of optimizer on NPU.

    Args:
        opt: optimizer instance.
        is_distributed: boolean, whether in distributed learning.
        mix_precision: boolean, whether to use mix precisioin.
        loss_scale: str, specifying the strategy to apply loss scaling. 
            Possible choices could be `off`: do not use loss scaling. 
            `d`: dynamic loss scaling, and `f*`: fixed loss scaling,
            where `*` can be converted to an integer that specifies the 
            scale factor, `2^(int(*))`.
    
    Returns:
        An optimizer instance.
    """
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from .loss_scaling import npu_loss_scale_optimizer
    if is_distributed:
        opt = NPUDistributedOptimizer(opt)
    if mix_precision:
        opt = npu_loss_scale_optimizer(opt, loss_scale, is_distributed)
    return opt