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
import tensorflow as tf

from src.utils.logger import logger
from src.utils.klass import get_subclass_given_name


def get_loss(loss_type, pred, groundtruth, weight_map=None, **kwargs):
    """Get the corresponding loss tensor given the loss type and the data pair.

    Args:
        loss_type: str, type of loss class.
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
            tensor, whose shape is the same with the pred and groundtruth.
    """
    try:
        klass = get_subclass_given_name(BaseLoss, loss_type)
    except IndexError:
        raise ValueError(f'Cannot find loss type {loss_type}.')
    return klass()(pred, groundtruth, weight_map=None, **kwargs)


class BaseLoss(object):
    def __call__(self, pred, grountruth, **kwargs):
        raise NotImplementedError


class L1Loss(BaseLoss):
    """Pixelwise L1-loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, weight_map=None, **kwargs):
        loss = tf.abs(pred - groundtruth)
        if weight_map is not None:
            loss = loss * weight_map
        return loss


class MarginalL1Loss(BaseLoss):
    """Pixelwise L1-loss with margins.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        eps: scalar, a small scalar to margin out the values that are too small.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, eps=1e-6, weight_map=None, **kwargs):
        
        loss = tf.maximum(tf.abs(pred - groundtruth), eps)
        if weight_map is not None:
            loss = loss * weight_map
        return loss


class L2Loss(BaseLoss):
    """Pixelwise L2-loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, weight_map=None, **kwargs):
        loss = tf.square(pred - groundtruth)
        if weight_map is not None:
            loss = loss * weight_map
        return loss


class HuberLoss(BaseLoss):
    """Pixelwise Huber loss, a.k.a. the smooth-l1 loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        delta: scalar, threshold to indicate where to change between l1 and l2. 
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, delta=1., weight_map=None, **kwargs):
        res = pred - groundtruth
        mask = tf.cast(tf.less(tf.abs(res), 1.), tf.float32)
        lesser_region = 0.5 * l2_loss(pred, groundtruth)
        greater_region = l1_loss(pred, groundtruth) - 0.5*delta**2
        loss = mask * lesser_region + (1. - mask) * greater_region
        if weight_map is not None:
            loss = loss * weight_map
        return loss


# Alias smooth l1 loss.
SmoothL1Loss = HuberLoss


class CharbonnierLoss(BaseLoss):
    """Pixelwise Charbonnier loss. A variant of L1-loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        eps: scalar, a small value to avoid inf or nan during sqrt.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, eps=1e-6, weight_map=None, **kwargs):
        
        loss = tf.sqrt((pred - groundtruth) ** 2 + eps)
        if weight_map is not None:
            loss = loss * weight_map
        return loss


class MSELoss(BaseLoss):
    """
    Pixelwise mse loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, weight_map=None, **kwargs):
        
        loss = tf.square(groundtruth - pred)
        if weight_map is not None:
            loss = loss * weight_map

        return loss


class FocalLoss(BaseLoss):
    """Pixelwise FocalLoss. See https://arxiv.org/pdf/1708.02002.pdf

    Args:s
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        alpha: scalar, a small balance value. Default 0.25 as in the paper.
        gamma: scalar, focusing parameter which is greater than 0. Default 2.
        eps: scalar, a small value to avoid nan when tf.log.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
        """
    def __call__(self, pred, groundtruth, alpha=0.25, gamma=2, eps=1e-6, 
                 weight_map=None, **kwargs):
        
        pt = tf.where(groundtruth, pred, 1.-pred)
        loss = - alpha * tf.pow(1. - pt, gamma) * tf.log(tf.maximum(pt, eps))
        if weight_map is not None:
            loss = loss * weight_map
        return loss


class CosineDistanceLoss(BaseLoss):
    """Pixelwise cosine distance loss.

    Args:
        pred: tensor, the predictions.
        groundtruth: tensor, the target tensor.
        axis: int, which axis to do the normalization.
        eps: scalar, a small value to avoid nan when tf.log.
        weight_map: tensor or None. If given, the loss will be weighted.

    Returns:
        tensor, whose shape is the same with the pred and groundtruth.
    """
    def __call__(self, pred, groundtruth, axis=-1, eps=1e-6, weight_map=None, 
                 **kwargs):
        
        prod = pred * groundtruth
        prod = tf.reduce_sum(prod, axis=axis, keepdims=True)
        pred_norm = tf.reduce_sum(tf.square(pred), axis=axis, keepdims=True)
        gt_norm = tf.reduce_sum(tf.square(groundtruth), axis=axis, keepdims=True)
        norm_scale = tf.sqrt(pred_norm * gt_norm + eps)
        loss = 1. - prod / norm_scale
        if weight_map is not None:
            loss = loss * weight_map
        return loss
