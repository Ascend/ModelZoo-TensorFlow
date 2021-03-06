# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# Copyright 2020 Huawei Technologies Co., Ltd
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

# for NPU
#import horovod.tensorflow as hvd
# for NPU
import tensorflow as tf

from model.unet3d import Builder
from model.losses import make_loss, eval_dice, total_dice
from dataset.data_loader import CLASSES

# for NPU
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_optimizer import NPUOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
# for NPU


def unet_3d(features, labels, mode, params):

    logits = Builder(n_classes=4, normalization=params.normalization, mode=mode)(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = tf.argmax(input=logits, axis=-1, output_type=tf.dtypes.int32)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={'predictions': tf.cast(prediction, tf.int8)})

    labels = tf.cast(labels, tf.float32)
    if not params.include_background:
        labels = labels[..., 1:]
        logits = logits[..., 1:]

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_acc = eval_dice(y_true=labels, y_pred=tf.round(logits))
        total_eval_acc = total_dice(tf.round(logits), labels)
        metrics = {CLASSES[i]: tf.metrics.mean(eval_acc[i]) for i in range(eval_acc.shape[-1])}
        metrics['WholeTumor'] = tf.metrics.mean(total_eval_acc)
        return tf.estimator.EstimatorSpec(mode=mode, loss=tf.reduce_mean(eval_acc),
                                          eval_metric_ops=metrics)

    loss = make_loss(params, y_pred=logits, y_true=labels)
    loss = tf.identity(loss, name="total_loss_ref")

    global_step = tf.compat.v1.train.get_or_create_global_step()

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate)
    # for NPU
    #optimizer = hvd.DistributedOptimizer(optimizer)
    if params.npu_loss_scale < 1:
        # disable npu_loss_scale
        optimizer = NPUDistributedOptimizer(optimizer)	
    else:
        # enable npu_dynamic_loss_scale
        if params.npu_loss_scale == 1:
            loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        # enable npu_static_loss_scale
        elif params.npu_loss_scale > 1:
            loss_scale_manager = FixedLossScaleManager(loss_scale=params.npu_loss_scale)
		
        if int(os.getenv('RANK_SIZE')) == 1:
            optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
        else:
            optimizer = NPUDistributedOptimizer(optimizer)
            optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)
    # for NPU

    # NGC has TF_ENABLE_AUTO_MIXED_PRECISION enabled by default. We cannot use
    # both graph_rewrite and envar, so if we're not in NGC we do graph_rewrite
    try:
        amp_envar = int(os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']) == 1
    except KeyError:
        amp_envar = False

    if params.use_amp and not amp_envar:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            optimizer,
            loss_scale='dynamic'
        )

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)
