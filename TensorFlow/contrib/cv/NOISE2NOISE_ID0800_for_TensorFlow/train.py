# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# ============================================================================
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
# ============================================================================
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

import tensorflow as tf
import numpy as np
from npu_bridge.npu_init import *
import dnnlib
import os
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset


class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x):
        (minval, maxval) = self.train_stddev_range
        shape = tf.shape(x)
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval / 255.0, maxval=maxval / 255.0)
        return x + tf.random_normal(shape) * rng_stddev

    def add_validation_noise_np(self, x):
        return x + np.random.normal(size=x.shape) * (self.validation_stddev / 255.0)


class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        return tf.random_poisson(chi_rng * (x + 0.5), shape=[]) / chi_rng - 0.5

    def add_validation_noise_np(self, x):
        chi = 30.0
        return np.random.poisson(chi * (x + 0.5)) / chi - 0.5


def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5 + np.cos(t * np.pi) / 2) ** 2
        return learning_rate * smooth
    return learning_rate


def train(
        submit_config: dnnlib.SubmitConfig,
        iteration_count: int,
        eval_interval: int,
        minibatch_size: int,
        learning_rate: float,
        ramp_down_perc: float,
        noise: dict,
        tf_config: dict,
        validation_config: dict,
        train_tfrecords: str,
        noise2noise: bool,
        is_distributed,
        is_loss_scale):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    if not isinstance(is_distributed, bool):
        is_distributed = is_distributed[0]
    if not isinstance(is_loss_scale, bool):
        is_loss_scale = is_loss_scale[0]

    # Create a run context (hides low level details, exposes simple API to manage the run)
    ctx = dnnlib.RunContext(submit_config, config)

    # Initialize TensorFlow graph and session using good default settings
    sess = tfutil.init_tf(tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf, is_distributed)

    # Construct the network using the Network helper class and a function defined in config.net_config
    with tf.device(None):
        net = tflib.Network(**config.net_config)

    # Optionally print layer information
    net.print_layers()

    print('Building TensorFlow graph...')

    with tf.name_scope('Inputs'), tf.device(None):
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])

        noisy_input, noisy_target, clean_target = dataset_iter.get_next()

    # Define the loss function using the Optimizer helper class, this will take care of multi GPU

    with tf.device(None):
        net_npu = net

        denoised = net_npu.get_output_for(noisy_input)

        if noise2noise:
            meansq_error = tf.reduce_mean(tf.square(noisy_target - denoised))
        else:
            meansq_error = tf.reduce_mean(tf.square(clean_target - denoised))

    # 设置反向梯度切分策略
    # if is_distributed:
    #     set_split_strategy_by_size([60, 40])

    opt = tf.train.AdamOptimizer(lrate_in)

    # if is_distributed:
    #     opt = NPUOptimizer(opt_tmp, loss_scale_manager, is_distributed=is_distributed,
    #                        is_loss_scale=is_loss_scale, is_tailing_optimization=is_tailing_optimization)

    if is_distributed:
        opt = npu_distributed_optimizer_wrapper(opt)
    if is_loss_scale:
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=500,
                                                               decr_every_n_nan_or_inf=2, incr_ratio=2.0,
                                                               decr_ratio=0.8)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=is_distributed)

    train_step = opt.minimize(meansq_error)
    lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")

    init = tf.global_variables_initializer()
    tfutil.run([init])

    if is_distributed:
        root_rank = 0
        bcast_op = tfutil.broadcast_global_variables(root_rank, index=int(os.getenv('RANK_ID')))
        tfutil.run([bcast_op])

    # Create a log file for Tensorboard
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.get_default_graph())

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=iteration_count)

    # The actual training loop
    for i in range(iteration_count):
        # Whether to stop the training or not should be asked from the context
        if ctx.should_stop():
            break

        if i != 0 and i % eval_interval == 0:
            # if True:

            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()

            # Evaluate 'x' to draw a batch of inputs
            [source_mb, target_mb, loss, l_s] = tfutil.run([noisy_input, clean_target, meansq_error, lossScale])
            denoised = net.run(source_mb)

            # save_image(submit_config, denoised[0], "img_{0}_y_pred.png".format(i))
            # save_image(submit_config, target_mb[0], "img_{0}_y.png".format(i))
            # save_image(submit_config, source_mb[0], "img_{0}_x_aug.png".format(i))

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)

            with npu_scope.without_npu_compile_scope():
                print(
                    'iter %-10d time %-12s sec/eval %-7.2f sec/iter %-7.3f maintenance %-6.3f loss %-7.6f loss_scale %-10d' % (
                        autosummary('Timing/iter', i),
                        dnnlib.util.format_time(autosummary('Timing/total_sec', time_total)),
                        autosummary('Timing/sec_per_eval', time_train),
                        autosummary('Timing/sec_per_iter', time_train / eval_interval),
                        autosummary('Timing/maintenance_sec', time_maintenance),
                        autosummary('Loss', loss),
                        autosummary('loss_scale', l_s)))

                dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=i, max_epoch=iteration_count)
            time_maintenance = ctx.get_last_update_interval() - time_train

        # autosummary("Loss", meansq_error)
        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        if (i + 1) % eval_interval == 0:
            lr = lrate

        tfutil.run([train_step], {lrate_in: lrate})

    saver = tf.train.Saver()
    tf.io.write_graph(sess.graph, submit_config.run_dir + '/ckpt_npu', 'graph.pbtxt', as_text=True)
    saver.save(sess=sess, save_path=submit_config.run_dir + "/ckpt_npu/model.ckpt")
    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))

    save_snapshot(submit_config, net, 'final')

    # Summary log and context should be closed at the end
    summary_log.close()
    sess.close()
    ctx.close()
