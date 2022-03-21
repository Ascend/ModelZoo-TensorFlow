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
from npu_bridge.npu_init import *

import tensorflow as tf

def _add_loss_summaries(total_loss, summaries):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
#     for l in losses + [total_loss]:
#         # Name each loss as '(raw)' and name the moving average version of the loss
#         # as the original loss name.
#         summaries.append(tf.summary.scalar(l.op.name + ' (raw)', l))
#         summaries.append(tf.summary.scalar(l.op.name, loss_averages.average(l)))

    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, summaries,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss, summaries)
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=65536, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    

#     # Add histograms for trainable variables.
#     if log_histograms:
#         for var in tf.trainable_variables():
#             summaries.append(tf.summary.histogram(var.op.name, var))

#     # Add histograms for gradients.
#     if log_histograms:
#         for grad, var in grads:
#             if grad is not None:
#                 summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op] + update_ops):
        train_op = tf.no_op(name='train')

    return train_op

