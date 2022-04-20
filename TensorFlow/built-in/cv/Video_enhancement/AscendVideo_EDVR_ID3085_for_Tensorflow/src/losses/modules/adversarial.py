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

import os
import numpy as np
import tensorflow as tf

from src.losses.modules.gan import get_gan
from src.utils.klass import get_subclass_given_name
from src.runner.common import name_space


def build_adversarial_loss(hq, gt, cfg):
    """hq and gt both in shape [b*num_out_frames, h, w, c]
    """
    num_out_frames = cfg.data.num_data_gt_frames
    adv_loss_type = cfg.loss.adversarial.loss_type
    try:
        loss_model = get_subclass_given_name(_BaseAdvLoss, 
                                             adv_loss_type)(cfg)
    except IndexError:
        logger.error(f'Cannot find adversarial loss type {adv_loss_type}.')
        raise ValueError()

    hr_shape = gt.get_shape().as_list()

    # Check whether 3D network is used and reshape the tensor to 4D or 5D.
    if '3D' in cfg.loss.adversarial.gan_type and len(hr_shape) == 4:
        gt = tf.reshape(gt, [-1, num_out_frames, *hr_shape[1:]])
        hq = tf.reshape(hq, [-1, num_out_frames, *hr_shape[1:]])
    elif (not '3D' in cfg.loss.adversarial.gan_type) and len(hr_shape) != 4:
        gt = tf.reshape(gt, [-1, *hr_shape[2:]])
        hq = tf.reshape(hq, [-1, *hr_shape[2:]])

    return loss_model(real=gt, fake=hq)


class _BaseAdvLoss:
    """Base adversarial loss class. 
    All the adv losses will be derived from the base class.

    After real data point and fake one forward through the discriminator,
    the logits will be used to calculate the losses.
    """
    def __init__(self, cfg):
        reduction = cfg.loss.adversarial.loss_reduction
        self.discriminator = get_gan(cfg)
        self.reduction = reduction
        self.cfg = cfg

    def __call__(self, real, fake):
        """
        Forward the real and fake sample through the discriminator and
        calculate the losses.

        Args:
            real: tensor, 4D or 5D tensor for real samples.
            fake: tensor, the same as real. Fake generated samples.
        
        Returns:
            real_loss: scalar tensor, loss for the real sample.
            fake_loss: scalar tensor, loss for the fake sample.
        """

        # Forward through the discriminators to get the logits.
        fake_logit = self.discriminator(fake)
        real_logit = self.discriminator(real)

        # Calculate the losses.
        real_loss, fake_loss = self.forward(real_logit, fake_logit)

        # Cast to fp32 before reduction in case of precision loss on Ascend.
        real_loss = tf.cast(real_loss, tf.float32)
        fake_loss = tf.cast(fake_loss, tf.float32)

        if self.reduction == 'mean':
            reduction_fn = tf.reduce_mean
        elif self.reduction == 'sum':
            reduction_fn = tf.reduce_sum
        else:
            raise NotImplementedError

        # Apply weights before output.
        real_loss = reduction_fn(real_loss) * self.cfg.loss.adversarial.loss_weight
        fake_loss = reduction_fn(fake_loss) * self.cfg.loss.adversarial.loss_weight

        name_space.add_to_collection(name_space.DiscriminatorLoss,
                                     'discriminator', real_loss)
        name_space.add_to_collection(name_space.GeneratorLoss,
                                     'adversarial', fake_loss)

        return real_loss, fake_loss

    def forward(self, real_logit, fake_logit):
        raise NotImplementedError


class VanillaAdvLoss(_BaseAdvLoss):
    """ 
    Vanialla adversarial loss, i.e.
    loss_d = E(log D) + E(log (1 - D(G)))
    loss_g = - E(log D(G))
    """
    def forward(self, real_logit, fake_logit):
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=fake_logit,
                            labels=tf.ones_like(fake_logit))
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=real_logit,
                            labels=tf.ones_like(real_logit)) \
                    + tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=fake_logit,
                            labels=tf.zeros_like(fake_logit))

        return real_loss, fake_loss


class HingeAdvLoss(_BaseAdvLoss):
    """ 
    Hinge adversarial loss, i.e.
    loss_d = E(max(0, 1 - D)) + E(max(0, 1 + D(G)))
    loss_g = - E(D(G))
    """
    def forward(self, real_logit, fake_logit):
        fake_loss = - fake_logit
        real_loss = tf.nn.relu(1.0 - real_logit) + tf.nn.relu(1.0 + fake_logit)
        return real_loss, fake_loss


class RSAdvLoss(VanillaAdvLoss):
    """ 
    Relativistic adversarial loss, i.e.
    loss_d = - E(log sigmoid(D - D(G)))
    loss_g = - E(log sigmoid(D(G) - D))
    """
    # Relativistic Standard GAN
    def forward(self, real_logit, fake_logit):
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_logit,
                    labels=tf.ones_like(fake_logit))
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=real_logit,
                    labels=tf.ones_like(real_logit))

        return real_loss, fake_loss

    def __call__(self, real, fake):
        fake_logit = self.discriminator(fake)
        real_logit = self.discriminator(real)
        real_loss, fake_loss = self.forward(real_logit - fake_logit,
                                            fake_logit - real_logit)

        real_loss = tf.cast(real_loss, tf.float32)
        fake_loss = tf.cast(fake_loss, tf.float32)

        if self.reduction == 'mean':
            reduction_fn = tf.reduce_mean
        elif self.reduction == 'sum':
            reduction_fn = tf.reduce_sum
        else:
            raise NotImplementedError

        real_loss = reduction_fn(real_loss)
        fake_loss = reduction_fn(fake_loss) * self.cfg.loss.adversarial.loss_weight

        name_space.add_to_collection(name_space.DiscriminatorLoss,
                                     'discriminator', real_loss)
        name_space.add_to_collection(name_space.GeneratorLoss,
                                     'adversarial', fake_loss)

        return real_loss, fake_loss


class RaSAdvLoss(VanillaAdvLoss):
    """ 
    Relativistic adversarial loss, i.e.
    loss_d = - E(log sigmoid(D - E(D(G)))) + E(log sigmoid(D(G) - E(D)))
    loss_g = - E(log sigmoid(D(G) - E(D))) + E(log sigmoid(D - E(D(G))))
    """
    # Relativistic Average GAN
    def forward(self, real_logit, fake_logit):
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                            labels=tf.ones_like(fake_logit)) \
                    + tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                              labels=tf.zeros_like(fake_logit))
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                            labels=tf.ones_like(real_logit)) \
                    + tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                              labels=tf.zeros_like(fake_logit))

        return real_loss, fake_loss

    def __call__(self, real, fake):
        fake_logit = self.discriminator(fake)
        real_logit = self.discriminator(real)
        real_loss, fake_loss = self.forward(real_logit - tf.reduce_mean(fake_logit, axis=0, keepdims=True),
                                            fake_logit - tf.reduce_mean(real_logit, axis=0, keepdims=True))

        real_loss = tf.cast(real_loss, tf.float32)
        fake_loss = tf.cast(fake_loss, tf.float32)

        if self.reduction == 'mean':
            reduction_fn = tf.reduce_mean
        elif self.reduction == 'sum':
            reduction_fn = tf.reduce_sum
        else:
            raise NotImplementedError

        real_loss = reduction_fn(real_loss)
        fake_loss = reduction_fn(fake_loss) * self.cfg.loss.adversarial.loss_weight

        name_space.add_to_collection(name_space.DiscriminatorLoss,
                                     'discriminator', real_loss)
        name_space.add_to_collection(name_space.GeneratorLoss,
                                     'adversarial', fake_loss)

        return real_loss, fake_loss


class LSAdvLoss(_BaseAdvLoss):
    """ 
    Least-square adversarial loss, i.e.
    loss_d = 0.5 * E((D - 1)**2) + 0.5 * E(D(G)**2)
    loss_g = E((D(G) - 1)**2)
    """
    def forward(self, real_logit, fake_logit):
        fake_loss = tf.square(fake_logit - 1)
        real_loss = 0.5 * tf.square(real_logit - 1) + 0.5 * tf.square(fake_logit)

        return real_loss, fake_loss


class WGANLoss(_BaseAdvLoss):
    """ 
    Wesserstein adversarial loss, i.e.
    loss_d = E(D) + E(1 - D(G))
    loss_g = - E(D(G))
    """
    def forward(self, real_logit, fake_logit):
        fake_loss = - fake_logit
        real_loss = fake_logit - real_logit
        return real_loss, fake_loss


class WGAN_GP_Loss(WGANLoss):
    """ 
    Wesserstein adversarial loss with gradient panelty, i.e.
    loss_d = E(D) + E(1 - D(G)) + GP(D, D(G))
    loss_g = - E(D(G))
    """
    def gradient_penalty(self, real, fake):
        b = real.get_shape().as_list()[0]
        ndim = len(real.get_shape().as_list())
        shape = (b, ) + (1, ) * (ndim - 1)
        alpha = tf.random.uniform(shape=shape)
        interpolates = alpha * real + (1. - alpha) * fake

        interp_logit = self.discriminator(interpolates)
        grads = tf.gradients(
            interp_logit,
            xs=interpolates
        )
        gradient_penalty = tf.reduce_mean((grads-1.)**2)
        return gradient_penalty

    def __call__(self, real, fake):
        fake_logit = self.discriminator(fake)
        real_logit = self.discriminator(real)
        real_loss, fake_loss = self.forward(real_logit, fake_logit)
        grad_penalty = self.gradient_penalty(real, fake)

        real_loss = tf.cast(real_loss, tf.float32)
        fake_loss = tf.cast(fake_loss, tf.float32)
        grad_penalty = tf.cast(grad_penalty, tf.float32)

        if self.reduction == 'mean':
            reduction_fn = tf.reduce_mean
        elif self.reduction == 'sum':
            reduction_fn = tf.reduce_sum
        else:
            raise NotImplementedError

        real_loss = reduction_fn(real_loss) \
                    + grad_penalty * self.cfg.loss.adversarial.grad_penalty_weight
        fake_loss = reduction_fn(fake_loss) * self.cfg.loss.adversarial.loss_weight

        name_space.add_to_collection(name_space.DiscriminatorLoss,
                                     'discriminator', real_loss)
        name_space.add_to_collection(name_space.GeneratorLoss,
                                     'adversarial', fake_loss)

        return real_loss, fake_loss
