"""Utilities derived from the VAT code."""
from npu_bridge.npu_init import *

import tensorflow as tf


def generate_perturbation(x, logit, forward, epsilon, xi=1e-6):
    """Generate an adversarial perturbation.

    Args:
        x: Model inputs.
        logit: Original model output without perturbation.
        forward: Callable which computs logits given input.
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
        epsilon: Gradient multiplier.
        xi: Small constant.

    Returns:
        Aversarial perturbation to be applied to x.
    """
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(1):
        d = xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(tf.reduce_mean(dist), [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return epsilon * get_normalized_vector(d)


def kl_divergence_with_logit(q_logit, p_logit):
    """Compute the per-element KL-divergence of a batch."""
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_sum(q * logsoftmax(q_logit), 1)
    qlogp = tf.reduce_sum(q * logsoftmax(p_logit), 1)
    return qlogq - qlogp


def get_normalized_vector(d):
    """Normalize d by infinity and L2 norms."""
    d /= 1e-12 + tf.reduce_max(
        tf.abs(d), list(range(1, len(d.get_shape()))), keepdims=True
    )
    d /= tf.sqrt(
        1e-6
        + tf.reduce_sum(
            tf.pow(d, 2.0), list(range(1, len(d.get_shape()))), keepdims=True
        )
    )
    return d


def logsoftmax(x):
    """Compute log-domain softmax of logits."""
    xdev = x - tf.reduce_max(x, 1, keepdims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
    return lsm

