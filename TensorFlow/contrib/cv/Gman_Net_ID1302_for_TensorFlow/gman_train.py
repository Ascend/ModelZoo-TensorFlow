"""
train
"""
# coding=utf-8
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
from npu_bridge.npu_init import NPULossScaleOptimizer, npu_config_proto, RewriterConfig, \
    ExponentialUpdateLossScaleManager, FixedLossScaleManager
# from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import moxing as mx
import gman_flags as df
import os
import tensorflow as tf
from datetime import datetime
import time
import numpy as np

import gman_config as dc
import gman_constant as constant
from datasets import parse_record
import gman_log
import gman_model as model
import gman_net as net
import gman_tower as tower
import gman_learningrate as learning_rate

# from PerceNet import *
if not os.path.exists(df.FLAGS.train_dir):
    os.makedirs(df.FLAGS.train_dir)
    print('make dir: {}'.format(df.FLAGS.train_dir + '/'))
logger = gman_log.def_logger(df.FLAGS.train_dir + "/log.txt")
if df.FLAGS.train_restore:
    mx.file.copy_parallel(df.FLAGS.checkpoint_dir_obs, df.FLAGS.checkpoint_dir)


def train_load_previous_model(path, saver, sess, init=None):
    """

    Args:
        path:
        saver:
        sess:
        init:

    Returns:

    """
    gmean_ckpt = tf.train.get_checkpoint_state(path)
    if gmean_ckpt and gmean_ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, gmean_ckpt.model_checkpoint_path)
    else:
        sess.run(init)


def train(image_number, config):
    """

    Args:
        image_number:
        config:

    Returns:

    """
    logger.info("Training on: %s" % df.FLAGS.data_url)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # Calculate the learning rate schedule.
        if constant.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN < df.FLAGS.batch_size:
            raise RuntimeError(' NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN cannot smaller than batch_size!')
        decay_steps = int(constant.NUM_EPOCHS_PER_DECAY)

        initial_learning_rate = learning_rate.LearningRate(constant.INITIAL_LEARNING_RATE,
                                                           df.FLAGS.train_dir + df.FLAGS.train_learning_rate)
        lr = tf.train.exponential_decay(initial_learning_rate.load(),
                                        global_step,
                                        decay_steps,
                                        initial_learning_rate.decay_factor,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=100,
                                                               decr_every_n_nan_or_inf=2, decr_ratio=0.8)
        # loss_scale_manager = FixedLossScaleManager(5000)
        # opt_tmp = tf.train.MomentumOptimizer(lr, momentum=0.9)
        opt_tmp = tf.train.RMSPropOptimizer(lr)
        # opt_tmp = tf.train.GradientDescentOptimizer(lr)
        # opt_tmp = tf.train.AdamOptimizer(lr)
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
        # opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        # load data df.FLAGS.data_paths
        dataset = tf.data.TFRecordDataset([df.FLAGS.data_path], compression_type='ZLIB').repeat()
        dataset = dataset.map(parse_record, num_parallel_calls=8)
        dataset = dataset.batch(df.FLAGS.batch_size, drop_remainder=True).prefetch(16)
        iterator = tf.data.make_initializable_iterator(dataset)
        batch_queue = iterator.get_next('getnext')
        # batch_queue = di.input_get_queue_from_tfrecord(tf_record_path, df.FLAGS.batch_size,
        #                                                df.FLAGS.input_image_height, df.FLAGS.input_image_width)
        # Calculate the gradients for each model tower.
        # vgg_per = Vgg16()
        # tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            gman_model = model.Gman()
            gman_net = net.Net(gman_model)
            with tf.device('/cpu:0'):
                with tf.name_scope('%s_%d' % (constant.TOWER_NAME, 0)) as scope:
                    gman_tower = tower.Gmantower(gman_net, batch_queue, scope, opt)
                    summaries, loss = gman_tower.process()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        # grads = tower.Tower.average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(gman_tower.tower_grades, global_step=global_step)
        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(constant.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        # , variables_averages_op
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # train_op = tf.group(apply_gradient_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        lossT = tf.placeholder(tf.float32)
        summaries.append(tf.summary.scalar('AvgTrnLoss', lossT))
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # set precision mode allow_mix_precision allow_fp32_to_fp16 force_fp32
        custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
            'force_fp32')
        # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        # # dump path
        # custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes('/cache/saveModels/')
        # # set dump debug
        # custom_op.parameter_map['enable_dump_debug'].b = True
        # custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        #     '{"output":"/cache/saveModels","task_trace":"on"}')
        # # custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(
        # #     '/home/ma-user/modelarts/user-job-dir/code/fusion_switch.cfg')
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        sess = tf.Session(config=npu_config_proto(config_proto=sess_config))
        # Restore previous trained model
        if config[dc.CONFIG_TRAINING_TRAIN_RESTORE]:
            train_load_previous_model(df.FLAGS.train_dir, saver, sess)
        else:
            sess.run(init)
        # init iterator
        sess.run(iterator.initializer)
        summary_writer = tf.summary.FileWriter(df.FLAGS.train_dir, sess.graph)
        max_step = int((image_number / df.FLAGS.batch_size) * 10)
        # For each tf-record, we train them twice.
        avgloss = 0.0
        # current_learning_rate = 1e-2
        for step in range(max_step):
            start_time = time.time()
            # if step != 0 and (step % 1000 == 0 or (step + 1) == max_step):
            _, loss_value, current_learning_rate = sess.run([train_op, loss, lr])
            # else:
            #     _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            avgloss = avgloss + loss_value
            if step % 100 == 100 - 1:
                num_examples_per_step = df.FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = ('%s: step: %d to %d, Avgloss = %.6f (%.1f examples/sec; %.3f '
                              's) Lr=%.6f')
                logger.info(format_str % (datetime.now(), step - 99, step, avgloss / 100.0,
                                          examples_per_sec, sec_per_batch, current_learning_rate))
                summary_str = sess.run(summary_op, feed_dict={lossT: avgloss / 100.0})
                summary_writer.add_summary(summary_str, step)
                avgloss = 0.0

            # Save the model checkpoint periodically.
            if step != 0 and (step % 1000 == 0 or (step + 1) == max_step):
                checkpoint_path = os.path.join(df.FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                initial_learning_rate.save(current_learning_rate)
        sess.close()
    logger.info("=========================================================================================")
    # total time: 101 min
    # copy results to obs
    mx.file.copy_parallel('/cache/saveModels', df.FLAGS.train_url)
    print('copy saved model to obs: {}.'.format(df.FLAGS.train_url))


if __name__ == '__main__':
    # ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_1116
    image_num = 140000
    config_file = dc.config_load_config()
    train(image_num, config_file)
